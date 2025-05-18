import subprocess
import os
from dotenv import load_dotenv
load_dotenv()

g_BUCKET                = os.getenv("BUCKET", "")
g_FOLDER                = os.getenv("FOLDER", "")
g_TASK_NAME             = os.getenv("TASK_NAME", "")         

STATE_FILE = "state.txt"

cmd = f'rclone cat r2:{g_BUCKET}/{g_FOLDER + "/" + g_TASK_NAME + "/" + STATE_FILE}'
print("\n\nExecute: " + cmd, flush = True)    
try:                                      
    subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE)
except subprocess.CalledProcessError as e:
    print(f"The error message: {e}")

cmd = f'rclone lsf r2:{g_BUCKET}/{g_FOLDER + "/" + g_TASK_NAME} -R'
print("\n\nExecute: " + cmd, flush = True)    
try:                                      
    subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE)
except subprocess.CalledProcessError as e:
    print(f"The error message: {e}")
