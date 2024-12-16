from flask import Flask, request, jsonify, send_file
import os
from PIL import Image
import uuid
from threading import Thread
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
import time
import numpy as np
import json
import shutil
import warnings
import tempfile
from storage import get_storage_provider

# Ignore xFormers warnings
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2.layers")

app = Flask(__name__)

# Constants
TEMP_DIR = tempfile.gettempdir()  # Use system temp directory
IMAGES_DIR = os.path.join(TEMP_DIR, "input_images")  # For temporary uploaded images
TASKS_DIR = os.path.join(TEMP_DIR, "output_tasks")   # For temporary task-specific data
TASK_STATUS = {}  # Store task status
MAX_SEED = 2**32 - 1

# Initialize storage provider
storage = get_storage_provider()

# Initialize the pipeline
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Create directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(TASKS_DIR, exist_ok=True)

def get_task_dir(task_id: str) -> str:
    """Get task-specific directory and create if it doesn't exist"""
    task_dir = os.path.join(TASKS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    return task_dir

def save_task_metadata(task_id: str, metadata: dict):
    """Save task metadata to task directory and S3"""
    task_dir = get_task_dir(task_id)
    metadata_path = os.path.join(task_dir, 'metadata.json')
    
    # Save locally first
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    # Upload to S3
    s3_path = f"tasks/{task_id}/metadata.json"
    storage.upload_file(metadata_path, s3_path)

def process_task(task_id: str, image_path: str, params: dict):
    """Background task processor in a different thread with given task_id"""
    task_dir = get_task_dir(task_id)
    
    try:
        TASK_STATUS[task_id]['status'] = 'processing'
        TASK_STATUS[task_id]['progress'] = 10
        
        # Copy input image to task directory
        task_image_path = os.path.join(task_dir, 'input.png')
        shutil.copy2(image_path, task_image_path)
        
        # Save task parameters and upload to S3
        params_path = os.path.join(task_dir, 'params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f)
        storage.upload_file(params_path, f"tasks/{task_id}/params.json")
        
        # Load and process image
        image = Image.open(task_image_path)
        
        # Run the pipeline with all parameters
        outputs = pipeline.run(
            image,
            seed=params.get('geometry_seed', 42),
            formats=["gaussian", "mesh"],
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": params.get('sparse_structure_steps', 12),
                "cfg_strength": params.get('sparse_structure_strength', 7.5),
            },
            slat_sampler_params={
                "steps": params.get('slat_steps', 12),
                "cfg_strength": params.get('slat_strength', 3.0),
            },
        )
        
        TASK_STATUS[task_id]['progress'] = 70
        
        # Extract GLB with texture settings
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,  # Fixed ratio as per example
            texture_size=1024,  # Fixed size as per example
            verbose=False
        )
        
        # Save GLB locally first
        glb_path = os.path.join(task_dir, 'model.glb')
        glb.export(glb_path)
        
        # Upload GLB to S3
        s3_path = f"tasks/{task_id}/model.glb"
        s3_url = storage.upload_file(glb_path, s3_path)
        
        # Update task status with S3 URL
        TASK_STATUS[task_id]['status'] = 'completed'
        TASK_STATUS[task_id]['progress'] = 100
        TASK_STATUS[task_id]['output'] = {
            'model': s3_url  # Store the S3 URL directly
        }
        
        # Save final status to task directory and S3
        save_task_metadata(task_id, TASK_STATUS[task_id])
        
        # Clean up local files
        shutil.rmtree(task_dir)
        
    except Exception as e:
        TASK_STATUS[task_id]['status'] = 'failed'
        TASK_STATUS[task_id]['error'] = str(e)
        save_task_metadata(task_id, TASK_STATUS[task_id])
        # Clean up local files
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)

@app.route('/trellis/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({
            'code': 2003,
            'data': {
                'message': 'No file provided'
            }
        }), 400
    
    try:
        file = request.files['file']
        
        # Try to open and validate image
        try:
            image = Image.open(file.stream)
            
            # Check if format is supported
            if image.format.lower() not in ['jpeg', 'jpg', 'png']:
                return jsonify({
                    'code': 2004,
                    'data': {
                        'message': 'Unsupported file type. Only JPEG and PNG are supported'
                    }
                }), 400
                
            image.verify()  # Verify it's actually an image
            file.stream.seek(0)  # Reset file pointer after verify
            image = Image.open(file.stream)  # Reopen for actual processing
            
        except Exception:
            return jsonify({
                'code': 2004,
                'data': {
                    'message': 'Invalid image file'
                }
            }), 400
            
        # Get image info
        width, height = image.size
        file.stream.seek(0)  # Reset file pointer after reading
        file_content = file.read()
        file_size = len(file_content)
        
        # Generate token and save file
        image_token = str(uuid.uuid4())
        temp_path = os.path.join(IMAGES_DIR, f"{image_token}.png")
        
        # Save file locally first
        with open(temp_path, 'wb') as f:
            f.write(file_content)
            
        # Upload to S3
        s3_path = f"images/{image_token}.png"
        storage.upload_file(temp_path, s3_path)
        
        # Clean up local file
        os.remove(temp_path)
        
        return jsonify({
            'code': 0,
            'data': {
                'message': 'Image uploaded successfully',
                'image_token': image_token,
                'type': 'image',
                'size': file_size,
                'width': width,
                'height': height
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'code': 500,
            'data': {
                'message': f'Internal server error: {str(e)}'
            }
        }), 500

@app.route('/trellis/task', methods=['POST'])
def create_task():
    """Create processing task endpoint
    accept payload like:
    {
        "type": "image_to_model",
        "model_version": "default",
        "file": {
            "type": "image",
            "file_token": "your_image_token"
        },
        "face_limit": 10000,
        "texture": true,
        "pbr": true,
        "texture_seed": 0,
        "geometry_seed": 0,
        "sparse_structure_steps": 12,      # Optional: Number of steps for sparse structure generation (default: 12)
        "sparse_structure_strength": 7.5,   # Optional: Guidance strength for sparse structure (default: 7.5)
        "slat_steps": 12,                  # Optional: Number of steps for SLAT generation (default: 12)
        "slat_strength": 3.0               # Optional: Guidance strength for SLAT (default: 3.0)
    }
    """
    data = request.json
    if not data:
        return jsonify({
            'code': 2002,
            'data': {
                'message': 'Invalid request format'
            }
        }), 400
        
    # Validate payload structure
    if data.get('type') != 'image_to_model':
        return jsonify({
            'code': 2002,
            'data': {
                'message': 'Invalid task type'
            }
        }), 400
        
    if not data.get('file'):
        return jsonify({
            'code': 2003,
            'data': {
                'message': 'File information missing'
            }
        }), 400
    
    if not data['file'].get('type') not in ['png', 'jpeg', 'jpg']:
        return jsonify({
            'code': 2004,
            'data': {
                'message': 'Unsupported file type'
            }
        }), 400
        
    # Get file token (equivalent to our image_id)
    file_token = data['file'].get('file_token')
    if not file_token:
        return jsonify({
            'code': 2003,
            'data': {
                'message': 'File token missing'
            }
        }), 400
    
    # Check if image exists
    image_path = os.path.join(IMAGES_DIR, f"{file_token}.png")
    if not os.path.exists(image_path):
        return jsonify({
            'code': 2003,
            'data': {
                'message': 'Image not found'
            }
        }), 404
    
    # Validate and extract parameters with defaults
    try:
        face_limit = int(data.get('face_limit', 10000))
        if face_limit <= 0:
            raise ValueError("face_limit must be positive")
            
        texture_seed = int(data.get('texture_seed', 0))
        geometry_seed = int(data.get('geometry_seed', 0))
        
        if texture_seed < 0 or texture_seed > MAX_SEED:
            raise ValueError(f"texture_seed must be between 0 and {MAX_SEED}")
        if geometry_seed < 0 or geometry_seed > MAX_SEED:
            raise ValueError(f"geometry_seed must be between 0 and {MAX_SEED}")
            
        # Validate sampling parameters
        sparse_structure_steps = int(data.get('sparse_structure_steps', 20))
        sparse_structure_strength = float(data.get('sparse_structure_strength', 7.5))
        slat_steps = int(data.get('slat_steps', 20))
        slat_strength = float(data.get('slat_strength', 3.0))
        
        if sparse_structure_steps < 1:
            raise ValueError("sparse_structure_steps must be positive")
        if slat_steps < 1:
            raise ValueError("slat_steps must be positive")
        if sparse_structure_strength <= 0:
            raise ValueError("sparse_structure_strength must be positive")
        if slat_strength <= 0:
            raise ValueError("slat_strength must be positive")
            
    except (ValueError, TypeError) as e:
        return jsonify({
            'code': 2005,
            'data': {
                'message': f'Invalid parameter value: {str(e)}'
            }
        }), 400
    
    # Extract parameters
    params = {
        'file_token': file_token,
        'model_version': data.get('model_version', 'default'),
        'face_limit': face_limit,
        'texture': bool(data.get('texture', True)),
        'pbr': bool(data.get('pbr', True)),
        'texture_seed': texture_seed,
        'geometry_seed': geometry_seed,
        'sparse_structure_steps': sparse_structure_steps,
        'sparse_structure_strength': sparse_structure_strength,
        'slat_steps': slat_steps,
        'slat_strength': slat_strength
    }
    
    # Create task
    task_id = str(uuid.uuid4())
    TASK_STATUS[task_id] = {
        'status': 'pending',
        'progress': 0,
        'created_at': time.time(),
        'params': params
    }
    
    # Start background processing
    thread = Thread(target=process_task, args=(task_id, image_path, params))
    thread.start()
    
    return jsonify({
        'code': 0,
        'data': {
            'message': 'Task created successfully',
            'task_id': task_id,
            'status': 'pending',
            'params': params
        }
    }), 200

@app.route('/trellis/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get task status endpoint
    
    Response format:
    {
        "code": 0,
        "data": {
            "task_id": "unique-uuid",
            "type": "image_to_model",
            "status": "running/success/failed/etc",
            "input": {
                "file_token": "image-token",
                "model_version": "version",
                "face_limit": 10000,
                "texture": true,
                "pbr": true,
                "texture_seed": 0,
                "geometry_seed": 0
            },
            "output": {},  # Will contain URLs when completed
            "progress": 0-100,
            "create_time": timestamp
        }
    }
    """
    if task_id not in TASK_STATUS:
        return jsonify({
            'code': 2001,
            'data': {
                'message': 'Task not found'
            }
        }), 404
    
    status_data = TASK_STATUS[task_id].copy()
    
    # Map internal status to API status
    status_mapping = {
        'pending': 'queued',
        'processing': 'running',
        'completed': 'success',
        'failed': 'failed'
    }
    
    # Generate full URL using request's host if task is completed
    if status_data['status'] == 'completed' and 'output' in status_data:
        model_path = status_data['output']['model']
        # Use request's host and scheme for URL
        base_url = request.host_url.rstrip('/')
        status_data['output']['model'] = f"{base_url}/files/{model_path}"
    
    # Prepare response data
    response_data = {
        'message': f"Task is {status_mapping.get(status_data['status'], 'unknown')}",
        'task_id': task_id,
        'type': 'image_to_model',
        'status': status_mapping.get(status_data['status'], 'unknown'),
        'input': status_data['params'],
        'output': status_data.get('output', {}),
        'progress': status_data['progress'],
        'create_time': int(status_data['created_at'])
    }
    
    # Add error information if task failed
    if status_data['status'] == 'failed':
        response_data['error'] = status_data.get('error', 'Unknown error')
        response_data['message'] = f"Task failed: {status_data.get('error', 'Unknown error')}"
    
    return jsonify({
        'code': 0,
        'data': response_data
    }), 200

# Update cleanup function to clean S3 objects
def cleanup_temp_files():
    """Clean up files older than 1 hour from S3"""
    # Implementation depends on your S3 cleanup requirements
    pass

# Add periodic cleanup (every hour)
def start_cleanup_scheduler():
    while True:
        cleanup_temp_files()
        time.sleep(3600)  # Sleep for 1 hour

# Start cleanup thread when app starts
cleanup_thread = Thread(target=start_cleanup_scheduler, daemon=True)
cleanup_thread.start()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
