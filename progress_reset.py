import subprocess
import os
from dotenv import load_dotenv
load_dotenv()

# Warning: Executing this code will reset the task by removing all the state file, checkpoints and the final model stored in the cloud.

g_BUCKET                = os.getenv("BUCKET", "")
g_FOLDER                = os.getenv("FOLDER", "")
g_TASK_NAME             = os.getenv("TASK_NAME", "")         

cmd = f'rclone delete r2:{g_BUCKET}/{g_FOLDER + "/" + g_TASK_NAME}'
print("Execute: " + cmd, flush = True)    
try:                                      
    subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE)
except subprocess.CalledProcessError as e:
    print(f"The error message: {e}")
