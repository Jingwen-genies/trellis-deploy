import requests
from pathlib import Path
import os
import time

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
    
    # Use stream=True for large files
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'model/gltf-binary' not in content_type and 'application/octet-stream' not in content_type:
            raise Exception(f"Unexpected content type: {content_type}")
            
        # Get content length if available
        total_size = int(response.headers.get('content-length', 0))
        
        with open(out_file_path, "wb") as f:
            if total_size == 0:  # No content length header
                f.write(response.content)
            else:
                # Stream the download in chunks
                chunk_size = 8192
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
        
        # Verify file size
        if total_size > 0:
            actual_size = out_file_path.stat().st_size
            if actual_size != total_size:
                raise Exception(f"Downloaded file size ({actual_size}) doesn't match expected size ({total_size})")
                
        return out_file_path
    else:
        raise Exception(f"Failed to download model from {url}: {response.status_code}")

if __name__ == "__main__":
    # Check if AWS credentials are set
    required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'S3_BUCKET']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Test the flow
    image_token = upload_image("assets/example_image/T.png")
    print(f"Image uploaded, token: {image_token}")

    task_id = submit_image_to_model(
         image_token, model_version="Trellis-image-large", texture_seed=1, geometry_seed=1
    )
    print(f"Task created, ID: {task_id}")

    # Poll for task completion
    while True:
        data = get_task_status(task_id)
        print(f"Task status: {data.get('status')}, progress: {data.get('progress')}%")
        
        if data.get('status') == 'success' and 'model' in data.get('output', {}):
            url = data['output']['model']
            output_path = Path("./download_result.glb")
            download_model(url, output_path)
            print(f"Model downloaded to {output_path}")
            break
        elif data.get('status') == 'failed':
            print(f"Task failed: {data.get('error', 'Unknown error')}")
            break
            
        time.sleep(5)  # Wait 5 seconds before polling again

