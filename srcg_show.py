from salad_cloud_sdk import SaladCloudSdk
import os
from dotenv import load_dotenv
load_dotenv()

SALAD_API_KEY = os.getenv("SALAD_API_KEY","")
ORGANIZATION_NAME = os.getenv("ORGANIZATION_NAME","")
PROJECT_NAME = os.getenv("PROJECT_NAME","")
CONTAINER_GROUP_NAME = os.getenv("CONTAINER_GROUP_NAME","")

sdk = SaladCloudSdk(
    api_key=SALAD_API_KEY, 
    timeout=10000
)

result = sdk.container_groups.get_container_group(
    organization_name=ORGANIZATION_NAME,
    project_name=PROJECT_NAME,
    container_group_name=CONTAINER_GROUP_NAME 
)
print(result)