import os
import shutil
import subprocess
import time
import json
from datetime import datetime, timezone
import threading
import queue
import speedtest
from salad_cloud_sdk import SaladCloudSdk
from pythonping import ping
import requests
from dotenv import load_dotenv
load_dotenv()

# Assigned automatically while running on SaladCloud
SALAD_MACHINE_ID =  os.getenv("SALAD_MACHINE_ID", "LOCAL")

# Access to the Cloudflare R2 
CLOUDFLARE_ENDPOINT_URL = os.getenv("CLOUDFLARE_ENDPOINT_URL", "")
CLOUDFLARE_REGION       = os.getenv("CLOUDFLARE_REGION", "")
CLOUDFLARE_ID           = os.getenv("CLOUDFLARE_ID", "")
CLOUDFLARE_KEY          = os.getenv("CLOUDFLARE_KEY", "")
g_BUCKET                = os.getenv("BUCKET", "")
g_FOLDER                = os.getenv("FOLDER", "")

# Download gated models from Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Call the SaladCloud API to shutdown the instance when the training is completed
SALAD_API_KEY        = os.getenv("SALAD_API_KEY","")
ORGANIZATION_NAME    = os.getenv("ORGANIZATION_NAME","")
PROJECT_NAME         = os.getenv("PROJECT_NAME","")

g_CONTAINER_GROUP_NAME = os.getenv("CONTAINER_GROUP_NAME","") # Define the container group name on SaladCloud
g_TASK_NAME            = os.getenv("TASK_NAME", "")           # Define the sub-folder in Cloudflare R2

# For node filtering: Network performance, supported CUDA Version and GPU VRAM available
g_MODEL                = os.getenv("MODEL", "")
g_EPOCHS               = int(os.getenv("EPOCHS", ""))
g_BATCH_SIZE           = int(os.getenv("BATCH_SIZE", ""))
g_SAVING_STEPS         = int(os.getenv("SAVING_STEPS", ""))
g_SEED                 = int(os.getenv("SEED", "42"))
g_DLSPEED              = int(os.getenv("DLSPEED", "50")) # Mbps
g_ULSPEED              = int(os.getenv("ULSPEED", "20")) # Mbps
g_CUDA_RT_VERSION      = float(os.getenv("CUDA_RT_VERSION", "12.6"))   # The CUDA version used by the container image
g_VRAM_AVAILABLE       = int(os.getenv("VRAM_AVAILABLE","22000"))      # 22000 MiB
g_MAX_NO_RESPONSE_TIME = int(os.getenv("MAX_NO_RESPONSE_TIME","3600")) # No response from the trainer (acting as Health Check), including model downloading and checkpoint saving

g_TASK_CREATION_TIME   =  os.getenv("TASK_CREATION_TIME", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))

# Create a clean folder for the task (remove if it already exists)
if os.path.exists(g_TASK_NAME):
    shutil.rmtree(g_TASK_NAME) 
os.makedirs(g_TASK_NAME)

STATE_FILE = "state.txt"
LOCAL_STATE_FILE = os.path.join(g_TASK_NAME, STATE_FILE)
REMOTE_STATE_FILE = g_FOLDER + "/" + g_TASK_NAME + "/" + STATE_FILE

g_State = {}

# Queue for signaling from the training thread to the upload thread when a checkpoint is created
upload_task_queue = queue.Queue()

######################################## SaladCloud

# Stop the SRCG (Single-Replica Container Group) when the training is completed
def Shutdown():
    local_run = True if SALAD_MACHINE_ID == "LOCAL" else False

    if (local_run):  # Run locally
        print("Call the exit(0) ......", flush=True)
        os._exit(0)
    else:            # Run on SaladCloud
        print("Call the SaladCloud API to shutdown ......", flush=True)
        # https://docs.salad.com/reference/saladcloud-api/container-groups/stop-container-group
        sdk = SaladCloudSdk(api_key=SALAD_API_KEY, timeout=10000)
        _ = sdk.container_groups.stop_container_group(
            organization_name = ORGANIZATION_NAME,
            project_name = PROJECT_NAME,
            container_group_name = g_CONTAINER_GROUP_NAME
        )
        time.sleep(10)
    
# Trigger node reallocation if a node is not suitable
# https://docs.salad.com/products/sce/container-groups/imds/imds-reallocate
def Reallocate(reason):
    local_run = True if SALAD_MACHINE_ID == "LOCAL" else False
    print(reason)

    if (local_run):  # Run locally
        print("Call the exit(1) ......", flush=True)
        os._exit(1)
    else:            # Run on SaladCloud
        print("Call the IMDS reallocate ......", flush=True)
        url = "http://169.254.169.254/v1/reallocate"
        headers = {'Content-Type': 'application/json',
                   'Metadata': 'true'}
        body = {"Reason": reason}
        _ = requests.post(url, headers=headers, json=body)
        time.sleep(10)

# Test network bandwdith
def network_test():
    print("Test the network speed ....................", flush=True)
    try:
        speed_test = speedtest.Speedtest()
        bserver = speed_test.get_best_server()
        dlspeed = int(speed_test.download() / (1000 * 1000))  # Convert to Mbps, not Mib
        ulspeed = int(speed_test.upload() / (1000 * 1000))  # Convert to Mbps, not Mib
        latency = bserver['latency'] # the latency to the test server
        country = bserver['country'] 
        location = bserver['name']
    except Exception as e:  
        # Some ISPs may block speed test traffic; in such cases, we fall back to the default network performance for the node.
        return "", "", 99999, 50, 20
    return country, location, latency, dlspeed, ulspeed    

# Test network latency
# Only the root user can run this code - no issue in containers
def ping_test(tCount=10):
    if tCount ==0:
        return 99999, 99999
    print("Test the latency ....................", flush=True)
    try:
        print("To: ec2.us-east-1.amazonaws.com")
        temp = ping('ec2.us-east-1.amazonaws.com', interval=1, count=tCount, verbose=True)
        latency_useast1 = temp.rtt_avg_ms      
        print("To: ec2.eu-central-1.amazonaws.com")  
        temp = ping('ec2.eu-central-1.amazonaws.com', interval=1, count=tCount,verbose=True)
        latency_eucentral1 = temp.rtt_avg_ms
    except Exception as e:  
        return 99999, 99999
    return latency_useast1, latency_eucentral1

# Read the supported CUDA RT Version
def Get_CUDA_Version():
    try:
        cmd = 'nvidia-smi'
        output = subprocess.check_output(cmd, shell=True, text=True)
        output = output.split("\n")[2]
        output = output.split("CUDA Version: ")[-1]
        version = float(output.split(" ")[0])
    except Exception as e: 
        return 0
    return version 

# Get the GPU info
def Get_GPU():
    try:
        result = {}
        cmd = 'nvidia-smi --query-gpu=gpu_name,memory.total,memory.used,memory.free,utilization.memory,temperature.gpu,utilization.gpu --format=csv,noheader'
        output = subprocess.check_output(cmd, shell=True, text=True)
        result['gpu'], result['vram_total'], result['vram_used'], result['vram_free'], result['vram_utilization'], result['gpu_temperature'], result['gpu_utilization'] = output.strip().split(', ')
    except Exception as e:
        return {}
    return result 

def Initial_Check():    
    local_run = True if SALAD_MACHINE_ID == "LOCAL" else False
    environment= {}

    if not local_run:       # Skip the initial checks if run locally    

        # Network test: bandwidth
        country, _, _, dlspeed, ulspeed = network_test() 
        print(f"The network test result - Country: {country}, DL Speed: {dlspeed} Mbps, UL Speed: {ulspeed} Mbps.", flush=True)
        if ulspeed < g_ULSPEED or dlspeed < g_DLSPEED: # Node filtering
            # Reallocate("poor network bandwith")
            return False, f"Poor network bandwith - Country: {country}, DL Speed: {dlspeed} Mbps, UL Speed: {ulspeed} Mbps"
         
        # Network test: latency to some locations
        latency_us, latency_eu = ping_test(tCount = 10) 
        print(f"The latency test result - To US: {latency_us} ms, To EU: {latency_eu} ms.", flush=True)
        
        # CUDA Version
        CUDA_version = Get_CUDA_Version()
        print(f"The supported CUDA RT version is {CUDA_version}.", flush=True)
        if CUDA_version == 0:
            return False, "Failed to read CUDA RT version"  
        if CUDA_version < g_CUDA_RT_VERSION:
            # Reallocate(f"The supported CUDA RT version is lower than {CUDA_version}")
            return False, f"The supported CUDA RT version is lower than {CUDA_version}"     

        # VRAM Usage
        GPU = Get_GPU()
        if GPU == {}:
            return False, "Failed to read GPU info"  
        VRAM_free = float(GPU['vram_free'].split(" ")[0])
        if VRAM_free < g_VRAM_AVAILABLE: # MiB
            #Reallocate(f"Low VRAM")
            return False, f"Low VRAM: {VRAM_free} MiB "   

        environment = { "Country":           country,
                        "DL Mbps":           dlspeed, 
                        "UL Mbps":           ulspeed,
                        "RTT to US-East ms": latency_us,
                        "RTT to EU-Cent ms": latency_eu,
                        "GPU":               GPU['gpu'],
                        "CUDA":              CUDA_version,
                        "VRAM_Total":        GPU['vram_total'],
                        "VRAM_Free":         GPU['vram_free'] }

    return True, environment

######################################## rclone setup

# Create the configuration file for rclone using the environment variables
# https://developers.cloudflare.com/r2/examples/rclone/
filename = os.path.expanduser("~")+"/.config/rclone/rclone.conf"
with open(filename,'w') as f:
    f.write("[r2]\n")
    f.write("type = s3\n")
    f.write("provider = Cloudflare\n")
    f.write("access_key_id = {}\n".format(CLOUDFLARE_ID))
    f.write("secret_access_key = {}\n".format(CLOUDFLARE_KEY))
    f.write("region = {}\n".format(CLOUDFLARE_REGION))
    f.write("endpoint = {}\n".format(CLOUDFLARE_ENDPOINT_URL))
    f.write("no_check_bucket = true")

######################################## Data Sync with Cloudfalre using rclone 

# For the download/upload throughput calculation
def Get_Folder_Size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):  # Make sure it's a file
                total_size += os.path.getsize(fp)
    return total_size

# upload_local_to_cloud: True for uploading, False for downloading
def Data_Sync(source, bucket, target, upload_local_to_cloud=True, chunk_size_mbype="10M", concurrency="10"):
    
    t_start  = time.perf_counter()
    if upload_local_to_cloud:  # Local to Cloud
        run = "Upload to Cloud: "
        cmd = f'rclone copyto {source} r2:{bucket}/{target} --s3-chunk-size={chunk_size_mbype} --transfers={concurrency} --ignore-times'
    else:                      #  Cloud to Local
        run = "Download from Cloud: "
        cmd = f'rclone copyto r2:{bucket}/{source} {target} --s3-chunk-size={chunk_size_mbype} --transfers={concurrency} --ignore-times'
    print(run + cmd, flush=True)         

    try:                                      
        subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"The error message: {e}")

    t_end = time.perf_counter()
    t_duration = round((t_end - t_start), 3) 

    if upload_local_to_cloud:           
        if not os.path.exists(source):
            file_size = 0
        elif os.path.isdir(source):
            file_size = round(Get_Folder_Size(source)/1_000_000, 3)  # MB, not MiB  
        else:
            file_size = round(os.path.getsize(source)/1_000_000, 3)  # MB, not MiB 
    else:                               
        if not os.path.exists(target):
            file_size = 0
        elif os.path.isdir(target):
            file_size = Get_Folder_Size(target)/1_000_000  # MB, not MiB  
        else:
            file_size = os.path.getsize(target)/1_000_000  # MB, not MiB
    
    throughput = round(file_size * 8/t_duration, 3)
    file_size = round(file_size, 6)
    if upload_local_to_cloud:
        print(f"File Size: {file_size} MB, UL Time: {t_duration} s, UL Throught: {throughput} Mbps", flush=True)
    else:
        print(f"File Size: {file_size} MB, DL Time: {t_duration} s, DL Throught: {throughput} Mbps", flush=True)
    
    # The success of upload to cloud can't be confirmed directly — estimate based on upload duration and throughput.
    # The success of download from cloud can be confirmed with the file size, upload duration and throughput. 
    return file_size, t_duration, throughput

######################################## Called by the training thread

# Resume the running state previously saved to the cloud
def Resume_From_Cloud():
    global g_State

    passed, result =  Initial_Check() # Taking less than 60 seconds
    
    # Download or initialize the state file
    temp = Data_Sync(g_FOLDER + "/" + g_TASK_NAME + "/" + STATE_FILE, g_BUCKET, LOCAL_STATE_FILE, upload_local_to_cloud=False)
    if temp[0] != 0: # Existing Task 
        with open(LOCAL_STATE_FILE,'r') as f:
            g_State = json.load(f)        
    else:            # Start fresh  
        g_State = { "task_name": g_TASK_NAME, "done": False, "previous_checkpoint": "", 
                    "task_creation_time": g_TASK_CREATION_TIME, 
                    "task_completion_time": "",
                    "task_duration": "", 
                    "task_history": [], 
                    "training_state": "" } 

    GO_LIVE_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if passed == True:
        g_State["task_history"].append((GO_LIVE_TIME, SALAD_MACHINE_ID, "node_accepted", result))     
    else:
        g_State["task_history"].append((GO_LIVE_TIME, SALAD_MACHINE_ID, "node_rejected", result))     
    
    # Resume training from the previous state while the intial check is passed, the task is not completed and there is a checkpoint 
    if  passed == True and g_State["done"] == False and g_State["previous_checkpoint"] != "": # Download the previous checkpoint to local
        source = g_FOLDER + "/" + g_TASK_NAME + "/" + g_State["previous_checkpoint"]
        target = g_TASK_NAME + "/" + g_State["previous_checkpoint"]
        tSize, tDuration, tThroughput = Data_Sync(source, g_BUCKET, target, upload_local_to_cloud=False) # If this step fails, the task should be aborted immediately.
        msg = f"File Size: {tSize} MB, DL Time: {tDuration} s, DL Throught: {tThroughput} Mbps"
        CKPT_DL_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        g_State["task_history"].append((CKPT_DL_TIME, SALAD_MACHINE_ID, "ckpt_downloaded", g_State["previous_checkpoint"], msg))       
        
    # Update the state file in the cloud with the new node
    with open(LOCAL_STATE_FILE, 'w') as f:
        json.dump(g_State, f, indent=2)
    Data_Sync(LOCAL_STATE_FILE, g_BUCKET, REMOTE_STATE_FILE, upload_local_to_cloud=True)
        
    print(g_State)

    # If the task has already been completed, shutdown the SRCG
    if g_State["done"] == True:
       print("The training has already been finished!", flush=True)
       Shutdown()

    # If the inital check failed, reallocate 
    if passed == False:
        Reallocate(result)

def Get_Checkpoint():
    return g_State["previous_checkpoint"]

def Notify_Uploader(state):
    upload_task_queue.put(state)

def Close_All():
    global g_State

    upload_task_queue.put(None) # Notify the uploader thread to terminate

    # Waiting for all checkpoints to upload
    while True:
        temp = upload_task_queue.qsize()
        if temp > 0:
            print(f"Checkpoints ({temp}) are being uploaded ...", flush=True)
            time.sleep(10)
        else:
            break

    # Save the model first    
    source = g_TASK_NAME + "/" + "final"
    target = g_FOLDER + "/" + g_TASK_NAME + "/" + "final"
    tSize, tDuration, tThroughput = Data_Sync(source, g_BUCKET, target, upload_local_to_cloud=True)
    msg = f"File Size: {tSize} MB, UL Time: {tDuration} s, UL Throught: {tThroughput} Mbps"

    # Then save the state file
    UPLOAD_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    g_State["task_history"].append((UPLOAD_TIME, SALAD_MACHINE_ID, 'model_uploaded', 'final', msg))  
    g_State['done'] = True
    g_State["task_completion_time"] = UPLOAD_TIME 

    start =  datetime.strptime(g_State["task_creation_time"],"%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(g_State["task_completion_time"],"%Y-%m-%d %H:%M:%S")
    g_State["task_duration"] = str(end - start)
    
    with open(LOCAL_STATE_FILE, 'w') as f:
        json.dump(g_State, f, indent=2)
    Data_Sync(LOCAL_STATE_FILE, g_BUCKET, REMOTE_STATE_FILE, upload_local_to_cloud=True)

    Shutdown() # Exit


######################################## The uploader thread

def Uploader(queue):
    global g_State
    
    no_response_time = 0

    while True:

        if queue.empty():
            time.sleep(10)
            no_response_time = no_response_time + 10
            if no_response_time > g_MAX_NO_RESPONSE_TIME:
                print("No response from the trainer thread for one hour: poor download throughput or node performance", flush=True)
                Reallocate("poor network throughput or node performance")
            continue

        no_response_time = 0 # Reset
        
        state = queue.get()  # May block here
        
        if state == None:     # Training completed
            print("The training is done and the uploader thread exits!", flush=True)
            queue.task_done() 
            break # exits

        elif state == 'start': # The model downloaded and training started
            TRAINING_START_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            end   =  datetime.strptime( TRAINING_START_TIME, "%Y-%m-%d %H:%M:%S" )
            start =  datetime.strptime( g_State["task_history"][-1][0], "%Y-%m-%d %H:%M:%S" )
            duration = str(end - start)
            msg = f"The dataset, tokenizer and model have been downloaded and loaded in {duration} (HH:MM:SS)" 

            if g_State["previous_checkpoint"] == "": # start fresh
                g_State["task_history"].append((TRAINING_START_TIME, SALAD_MACHINE_ID, "training_started", "fresh", msg))
            else:                                    # training resumed
                g_State["task_history"].append((TRAINING_START_TIME, SALAD_MACHINE_ID, "training_resumed", g_State["previous_checkpoint"], msg))

            with open(LOCAL_STATE_FILE, 'w') as f:
                json.dump(g_State, f, indent=2)
            Data_Sync(LOCAL_STATE_FILE, g_BUCKET, REMOTE_STATE_FILE, upload_local_to_cloud=True)

        else:                 # A checkpoint is saved
            global_step = state["global_step"]
            checkpoint = f"checkpoint-{global_step}"
        
            # Save the checkpoint first   
            source = g_TASK_NAME + "/" + checkpoint
            target = g_FOLDER + "/" + g_TASK_NAME + "/" + checkpoint
            tSize, tDuration, tThroughput  = Data_Sync(source, g_BUCKET, target, upload_local_to_cloud=True)
            msg = f"File Size: {tSize} MB, UL Time: {tDuration} s, UL Throught: {tThroughput} Mbps"

            # if the actual upload throughput is too low, trigger the IMDS reallocation
            
            if tThroughput < g_ULSPEED / 2:
                print(f"The actual upload throughput is only {tThroughput} Mbps, warning !!!")
        
            # Then save the state file
            CPT_UPLOAD_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            g_State["task_history"].append((CPT_UPLOAD_TIME, SALAD_MACHINE_ID, 'ckpt_uploaded', checkpoint, msg))  
            g_State["previous_checkpoint"] = checkpoint 
            g_State['training_state'] = state

            with open(LOCAL_STATE_FILE, 'w') as f:
                json.dump(g_State, f, indent=2)
            Data_Sync(LOCAL_STATE_FILE, g_BUCKET, REMOTE_STATE_FILE, upload_local_to_cloud=True)

        queue.task_done() # mainly for queue.join(), not for queue.qsize()

ul_thread = threading.Thread(target=Uploader, args=(upload_task_queue,))
ul_thread.start()
