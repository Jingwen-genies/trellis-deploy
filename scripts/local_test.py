import requests
from pathlib import Path
import os
import time
import boto3
from urllib.parse import urlparse

url_upload = "http://localhost:5000/trellis/upload"
url_task = "http://localhost:5000/trellis/task"
url_status = f"http://localhost:5000/trellis/task/{{}}"


def upload_image(file_path):
    """Upload image to API and return image token"""
    files = {'file': open(file_path, 'rb')}
    response = requests.post(url_upload, files=files)
    response_data = response.json()
    if response.status_code != 200 or response_data.get("code") != 0:
        raise Exception(f"Image upload failed: {response_data}")

    image_token = response_data["data"]["image_token"]
    return image_token

def submit_image_to_model(
          image_token, model_version, texture_seed, geometry_seed, face_limit=10000,
          sparse_structure_steps=20, sparse_structure_strength=7.5, slat_steps=20, slat_strength=3.0
    ):
        payload = {
            "type": "image_to_model",
            "model_version": model_version,
            "file": {
                "type": "image",
                "file_token": image_token
            },
            "face_limit": face_limit,
            "texture": True,
            "pbr": True,
            "texture_seed": texture_seed,
            "geometry_seed": geometry_seed,
            "sparse_structure_steps": sparse_structure_steps,
            "sparse_structure_strength": sparse_structure_strength,
            "slat_steps": slat_steps,
            "slat_strength": slat_strength,
        }
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(url_task, headers=headers, json=payload)
        print(response)
        response_data = response.json()
        if response.status_code != 200 or response_data.get("code") != 0:
            raise Exception(f"Task submission failed: {response_data}")
        task_id = response_data["data"]["task_id"]
        return task_id 


def get_task_status(task_id):
    url = url_status.format(task_id)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get('data', {})
        return data
    else:
        raise Exception(f"Failed to get task status: {response.text}")

def download_model(url, out_file_path):
    """Download the model file from S3 URL and save it to the specified path."""
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse S3 URL
    if not url.startswith('s3://'):
        raise ValueError(f"Expected S3 URL (s3://...), got: {url}")
    
    parsed_url = urlparse(url)
    bucket_name = parsed_url.netloc
    s3_key = parsed_url.path.lstrip('/')
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        print(f"Downloading from S3: bucket={bucket_name}, key={s3_key}")
        
        # Download the file
        s3_client.download_file(
            Bucket=bucket_name,
            Key=s3_key,
            Filename=str(out_file_path)
        )
        
        print(f"Successfully downloaded to: {out_file_path}")
        return out_file_path
        
    except Exception as e:
        raise Exception(f"Failed to download model from S3: {str(e)}")

if __name__ == "__main__":
    # Check AWS environment variables
    import os
    print("AWS Credentials Status:")
    print(f"Access Key: {'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not Set'}")
    print(f"Secret Key: {'Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not Set'}")
    print(f"Region: {os.getenv('AWS_REGION') or 'Not Set'}")
    # required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    # missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    # if missing_vars:
    #     raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")

    # New test script:
    print("\n=== Starting Full Test Flow ===")
    try:
        # # 1. Upload image
        # image_path = "./TRELLIS/assets/example_image/T.png"  # Adjust path as needed
        # print(f"\n1. Uploading image: {image_path}")
        # image_token = upload_image(image_path)
        # print(f"✓ Image uploaded successfully. Token: {image_token}")

        image_token = '381c81fd-c766-480b-83ee-70c0ecfe2788'

        # 2. Create task
        print("\n2. Creating task for 3D conversion")
        task_id = submit_image_to_model(
            image_token=image_token,
            model_version="Trellis-image-large",
            texture_seed=1,
            geometry_seed=1,
            face_limit=10000,
            sparse_structure_steps=12,
            sparse_structure_strength=7.5,
            slat_steps=12,
            slat_strength=3.0
        )
        print(f"✓ Task created successfully. ID: {task_id}")

        # 3. Poll for completion
        print("\n3. Waiting for task completion...")
        while True:
            data = get_task_status(task_id)
            status = data.get('status')
            progress = data.get('progress', 0)
            
            print(f"   Status: {status}, Progress: {progress}%")
            
            if status == 'success':
                print("✓ Task completed successfully!")
                print(data)
                break
            elif status == 'failed':
                error = data.get('error', 'Unknown error')
                print(f"✗ Task failed: {error}")
                raise Exception(f"Task processing failed: {error}")
            
            time.sleep(0.5)  # Wait 0.5 seconds before next check

        # 4. Download result
        print("\n4. Downloading result")
        if 'model' in data.get('output', {}):
            url = data['output']['model']
            output_path = Path("./result.glb")
            download_model(url, output_path)
            print(f"✓ Model downloaded successfully to: {output_path.absolute()}")
        else:
            print("✗ No model URL in response")
            raise Exception("No model URL in response")

        print("\n✓ Test completed successfully!")
        print("=== Test Flow Completed ===\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        print("=== Test Flow Failed ===\n")
        raise




