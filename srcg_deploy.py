
from salad_cloud_sdk import SaladCloudSdk
from salad_cloud_sdk.models import ContainerGroupCreationRequest
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

CLOUDFLARE_ENDPOINT_URL = os.getenv("CLOUDFLARE_ENDPOINT_URL", "")
CLOUDFLARE_REGION       = os.getenv("CLOUDFLARE_REGION", "")
CLOUDFLARE_ID           = os.getenv("CLOUDFLARE_ID", "")
CLOUDFLARE_KEY          = os.getenv("CLOUDFLARE_KEY", "")
BUCKET                  = os.getenv("BUCKET", "")
FOLDER                  = os.getenv("FOLDER", "")

HF_TOKEN             = os.getenv("HF_TOKEN","")

SALAD_API_KEY        = os.getenv("SALAD_API_KEY","")
ORGANIZATION_NAME    = os.getenv("ORGANIZATION_NAME","")
PROJECT_NAME         = os.getenv("PROJECT_NAME","")

g_CONTAINER_GROUP_NAME = os.getenv("CONTAINER_GROUP_NAME","") 
g_TASK_NAME            = os.getenv("TASK_NAME", "")         

g_MODEL            = os.getenv("MODEL", "")
g_EPOCHS           = int(os.getenv("EPOCHS", ""))
g_BATCH_SIZE       = int(os.getenv("BATCH_SIZE", ""))
g_SAVING_STEPS     = int(os.getenv("SAVING_STEPS", ""))
g_SEED             = int(os.getenv("SEED", "42"))
g_DLSPEED          = int(os.getenv("DLSPEED", "50")) # Mbps
g_ULSPEED          = int(os.getenv("ULSPEED", "20")) # Mbps
g_CUDA_RT_VERSION  = float(os.getenv("CUDA_RT_VERSION", "12.6")) 
g_VRAM_AVAILABLE   = int(os.getenv("VRAM_AVAILABLE","22000"))    # MiB
g_MAX_NO_RESPONSE_TIME = int(os.getenv("MAX_NO_RESPONSE_TIME","3600")) # Second

g_TASK_CREATION_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
g_IMAGE              = os.getenv("IMAGE", "")

########################################
########################################

sdk = SaladCloudSdk(
    api_key=SALAD_API_KEY, 
    timeout=10000
)

request_body = ContainerGroupCreationRequest(
   name=g_CONTAINER_GROUP_NAME,        
   display_name=g_CONTAINER_GROUP_NAME,
   container={
       "image": g_IMAGE,
       "resources": {
           "cpu": 8,
           "memory": 24576,
           "gpu_classes": ['ed563892-aacd-40f5-80b7-90c9be6c759b'],
           "storage_amount": 53687091200,
       },  # 50 GB
    
       # "command": ['sh', '-c', 'sleep infinity' ],
       "priority": "high",
       "environment_variables": {
               "CLOUDFLARE_ENDPOINT_URL": CLOUDFLARE_ENDPOINT_URL,
               "CLOUDFLARE_REGION": CLOUDFLARE_REGION,
               "CLOUDFLARE_ID": CLOUDFLARE_ID,
               "CLOUDFLARE_KEY": CLOUDFLARE_KEY,
               "BUCKET": BUCKET,
               "FOLDER": FOLDER,
               
               "HF_TOKEN": HF_TOKEN,      

               "SALAD_API_KEY": SALAD_API_KEY,
               "ORGANIZATION_NAME": ORGANIZATION_NAME,
               "PROJECT_NAME": PROJECT_NAME,
               
               "CONTAINER_GROUP_NAME": g_CONTAINER_GROUP_NAME,
               "TASK_NAME": g_TASK_NAME,
               
               "MODEL": g_MODEL,
               "EPOCHS": g_EPOCHS,
               "BATCH_SIZE": g_BATCH_SIZE,
               "SAVING_STEPS": g_SAVING_STEPS,
               "SEED": g_SEED,
               "ULSPEED": g_ULSPEED,                  
               "DLSPEED": g_DLSPEED,                  
               "CUDA_RT_VERSION": g_CUDA_RT_VERSION,                  
               "VRAM_AVAILABLE": g_VRAM_AVAILABLE,
               "MAX_NO_RESPONSE_TIME": g_MAX_NO_RESPONSE_TIME,
               
               "TASK_CREATION_TIME": g_TASK_CREATION_TIME
        }
   },
   autostart_policy=True,
   restart_policy="always",
   replicas=1,
   country_codes=[ "us" ]
)

print(request_body)


result = sdk.container_groups.create_container_group(
   request_body=request_body,
   organization_name=ORGANIZATION_NAME,
   project_name=PROJECT_NAME
)
print(result)

result = sdk.container_groups.get_container_group(
    organization_name=ORGANIZATION_NAME,
    project_name=PROJECT_NAME,
    container_group_name=g_CONTAINER_GROUP_NAME 
)
print(result)