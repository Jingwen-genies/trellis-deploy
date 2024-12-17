from flask import Flask, request, jsonify, send_file
import os
from PIL import Image
import uuid
from threading import Thread
import time
import json
import shutil
import warnings
import tempfile
import numpy as np
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import postprocessing_utils
from scripts.storage import get_storage_provider
from scripts.config import load_config
from scripts.logger_setup import setup_logging

# Initialize logger
logger = setup_logging()

# Ignore xFormers warnings
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2.layers")

class TrellisConfig:
    """Configuration management class"""
    def __init__(self):
        self.TEMP_DIR = tempfile.gettempdir()
        self.IMAGES_DIR = os.path.join(self.TEMP_DIR, "input_images")
        self.TASKS_DIR = os.path.join(self.TEMP_DIR, "output_tasks")
        self.MAX_SEED = 2**32 - 1
        
        # Load config file
        config = load_config("./config.yaml")
        self.s3_input_dir = config["storage"]['prefix'] + "input_images"
        self.s3_output_dir = config["storage"]['prefix'] + "output_tasks"
        
        # Create necessary directories
        os.makedirs(self.IMAGES_DIR, exist_ok=True)
        os.makedirs(self.TASKS_DIR, exist_ok=True)
        
        # Initialize storage
        self.storage = get_storage_provider(config["storage"])
        
        # Log configuration
        logger.info(f"TEMP_DIR: {self.TEMP_DIR}")
        logger.info(f"IMAGES_DIR: {self.IMAGES_DIR}")
        logger.info(f"TASKS_DIR: {self.TASKS_DIR}")
        logger.info(f"AWS_ACCESS_KEY_ID: {'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not Set'}")
        logger.info(f"AWS_SECRET_ACCESS_KEY: {'Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not Set'}")
        logger.info(f"AWS_REGION: {os.getenv('AWS_REGION') or 'Not Set'}")

class TrellisModel:
    """Model management class"""
    def __init__(self):
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline.cuda()
        
    def process_image(self, image: Image.Image, params: dict) -> dict:
        """Process an image with the Trellis pipeline"""
        return self.pipeline.run(
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

class TaskManager:
    """Task management class"""
    def __init__(self, config: TrellisConfig, model: TrellisModel):
        self.config = config
        self.model = model
        self.tasks = {}  # Store task status
        
    def get_task_dir(self, task_id: str) -> str:
        """Get task-specific directory and create if it doesn't exist"""
        task_dir = os.path.join(self.config.TASKS_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        return task_dir
        
    def save_task_metadata(self, task_id: str, metadata: dict):
        """Save task metadata to task directory and S3"""
        task_dir = self.get_task_dir(task_id)
        metadata_path = os.path.join(task_dir, 'metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        s3_path = f"{self.config.s3_output_dir}/{task_id}/metadata.json"
        return self.config.storage.upload_file(metadata_path, s3_path)
        
    def process_task(self, task_id: str, image_path: str, params: dict):
        """Process a task in background"""
        task_dir = self.get_task_dir(task_id)
        
        try:
            logger.info(f"Starting task processing for task_id: {task_id}")
            self.tasks[task_id]['status'] = 'processing'
            self.tasks[task_id]['progress'] = 10
            
            # Copy and upload input image
            task_image_path = os.path.join(task_dir, 'input.png')
            shutil.copy2(image_path, task_image_path)
            logger.debug(f"Copied input image to task directory: {task_image_path}")
            
            s3_input_path = f"{self.config.s3_output_dir}/{task_id}/input.png"
            self.config.storage.upload_file(task_image_path, s3_input_path)
            logger.debug(f"Uploaded input image to S3: {s3_input_path}")
            
            # Save and upload parameters
            params_path = os.path.join(task_dir, 'params.json')
            with open(params_path, 'w') as f:
                json.dump(params, f)
            s3_params_path = f"{self.config.s3_output_dir}/{task_id}/params.json"
            self.config.storage.upload_file(params_path, s3_params_path)
            logger.debug(f"Uploaded parameters to S3: {s3_params_path}")
            
            # Process image
            logger.info(f"Processing image for task {task_id}")
            image = Image.open(task_image_path)
            outputs = self.model.process_image(image, params)
            logger.info(f"Pipeline completed for task {task_id}")
            
            self.tasks[task_id]['progress'] = 70
            
            # Generate and save GLB
            logger.info(f"Generating GLB for task {task_id}")
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.95,
                texture_size=1024,
                verbose=False
            )
            
            glb_path = os.path.join(task_dir, 'model.glb')
            glb.export(glb_path)
            logger.debug(f"Saved GLB locally: {glb_path}")
            
            # Upload GLB
            s3_path = f"{self.config.s3_output_dir}/{task_id}/model.glb" # s3 path without bucket
            self.config.storage.upload_file(glb_path, s3_path)
            # Create direct S3 URL
            s3_url = f"s3://{self.config.storage.bucket_name}/{s3_path}"
            logger.info(f"S3 URL for GLB: {s3_url}")
            
            # Update task status
            self.tasks[task_id]['status'] = 'completed'
            self.tasks[task_id]['progress'] = 100
            self.tasks[task_id]['output'] = {'model': s3_url}
            
            self.save_task_metadata(task_id, self.tasks[task_id])
            logger.info(f"Task {task_id} completed successfully")
            
            # Cleanup
            shutil.rmtree(task_dir)
            logger.debug(f"Cleaned up task directory: {task_dir}")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
            self.tasks[task_id]['status'] = 'failed'
            self.tasks[task_id]['error'] = str(e)
            self.save_task_metadata(task_id, self.tasks[task_id])
            if os.path.exists(task_dir):
                shutil.rmtree(task_dir)
                logger.debug(f"Cleaned up task directory after failure: {task_dir}")
                
    def create_task(self, file_token: str, params: dict) -> str:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            'status': 'pending',
            'progress': 0,
            'created_at': time.time(),
            'params': params
        }
        return task_id
        
    def get_task_status(self, task_id: str) -> dict:
        """Get task status with proper mapping"""
        if task_id not in self.tasks:
            return None
            
        status_data = self.tasks[task_id].copy()
        status_mapping = {
            'pending': 'queued',
            'processing': 'running',
            'completed': 'success',
            'failed': 'failed'
        }
        
        return {
            'message': f"Task is {status_mapping.get(status_data['status'], 'unknown')}",
            'task_id': task_id,
            'type': 'image_to_model',
            'status': status_mapping.get(status_data['status'], 'unknown'),
            'input': status_data['params'],
            'output': status_data.get('output', {}),
            'progress': status_data['progress'],
            'create_time': int(status_data['created_at']),
            'error': status_data.get('error') if status_data['status'] == 'failed' else None
        }

class TrellisAPI:
    """Main API class"""
    def __init__(self):
        self.config = TrellisConfig()
        self.model = TrellisModel()
        self.task_manager = TaskManager(self.config, self.model)
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes"""
        self.app.route('/trellis/upload', methods=['POST'])(self.upload_image)
        self.app.route('/trellis/task', methods=['POST'])(self.create_task)
        self.app.route('/trellis/task/<task_id>', methods=['GET'])(self.get_task_status)
        
    def validate_image(self, file) -> tuple[Image.Image, str, int]:
        """Validate image file and return image object, format, and size"""
        try:
            image = Image.open(file.stream)
            if image.format.lower() not in ['jpeg', 'jpg', 'png']:
                raise ValueError(f"Unsupported file type: {image.format}")
            
            image.verify()
            file.stream.seek(0)
            image = Image.open(file.stream)
            file.stream.seek(0)
            content = file.read()
            
            return image, image.format.lower(), len(content)
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
            
    def upload_image(self):
        """Handle image upload"""
        if 'file' not in request.files:
            logger.warning("Upload attempt with no file provided")
            return jsonify({
                'code': 2003,
                'data': {'message': 'No file provided'}
            }), 400
        
        try:
            file = request.files['file']
            logger.info(f"Processing upload for file: {file.filename}")
            
            try:
                image, format_type, file_size = self.validate_image(file)
                width, height = image.size
                logger.debug(f"Image size: {width}x{height}, {file_size} bytes")
            except ValueError as e:
                logger.warning(str(e))
                return jsonify({
                    'code': 2004,
                    'data': {'message': str(e)}
                }), 400
            
            # Save and upload file
            image_token = str(uuid.uuid4())
            temp_path = os.path.join(self.config.IMAGES_DIR, f"{image_token}.png")
            
            with open(temp_path, 'wb') as f:
                file.stream.seek(0)
                f.write(file.read())
            logger.debug(f"Saved file locally: {temp_path}")
            
            s3_path = f"{self.config.s3_input_dir}/{image_token}.png"
            s3_url = self.config.storage.upload_file(temp_path, s3_path)
            logger.info(f"Uploaded file to S3: {s3_path}")
            
            os.remove(temp_path)
            logger.debug(f"Cleaned up temporary file: {temp_path}")
            
            return jsonify({
                'code': 0,
                'data': {
                    'message': 'Image uploaded successfully',
                    'image_token': image_token,
                    's3_url': s3_url,
                    'type': 'image',
                    'size': file_size,
                    'width': width,
                    'height': height
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}", exc_info=True)
            return jsonify({
                'code': 500,
                'data': {'message': f'Internal server error: {str(e)}'}
            }), 500
            
    def validate_task_params(self, data: dict) -> tuple[str, dict]:
        """Validate task parameters and return file_token and validated params"""
        if not data:
            raise ValueError("Invalid request format")
            
        if data.get('type') != 'image_to_model':
            raise ValueError("Invalid task type")
            
        if not data.get('file'):
            raise ValueError("File information missing")
            
        if not data['file'].get('type') not in ['png', 'jpeg', 'jpg']:
            raise ValueError("Unsupported file type")
            
        file_token = data['file'].get('file_token')
        if not file_token:
            raise ValueError("File token missing")
            
        # Validate numeric parameters
        face_limit = int(data.get('face_limit', 10000))
        if face_limit <= 0:
            raise ValueError("face_limit must be positive")
            
        texture_seed = int(data.get('texture_seed', 0))
        geometry_seed = int(data.get('geometry_seed', 0))
        
        if texture_seed < 0 or texture_seed > self.config.MAX_SEED:
            raise ValueError(f"texture_seed must be between 0 and {self.config.MAX_SEED}")
        if geometry_seed < 0 or geometry_seed > self.config.MAX_SEED:
            raise ValueError(f"geometry_seed must be between 0 and {self.config.MAX_SEED}")
            
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
        
        return file_token, params
            
    def create_task(self):
        """Handle task creation"""
        try:
            data = request.json
            logger.info(f"Creating new task with data: {json.dumps(data)}")
            
            try:
                file_token, params = self.validate_task_params(data)
            except ValueError as e:
                logger.warning(str(e))
                return jsonify({
                    'code': 2002,
                    'data': {'message': str(e)}
                }), 400
            
            # Download input image
            try:
                s3_path = f"{self.config.s3_input_dir}/{file_token}.png"
                download_path = os.path.join(self.config.IMAGES_DIR, f"{file_token}.png")
                self.config.storage.download_file(s3_path, download_path)
                
                with Image.open(download_path) as img:
                    img.verify()
            except Exception as e:
                logger.error(f"Failed to retrieve image: {str(e)}")
                return jsonify({
                    'code': 2003,
                    'data': {'message': f'Failed to retrieve image: {str(e)}'}
                }), 404
            
            # Create and start task
            task_id = self.task_manager.create_task(file_token, params)
            thread = Thread(
                target=self.task_manager.process_task,
                args=(task_id, download_path, params)
            )
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
            
        except Exception as e:
            logger.error(f"Task creation failed: {str(e)}", exc_info=True)
            return jsonify({
                'code': 500,
                'data': {'message': f'Internal server error: {str(e)}'}
            }), 500
            
    def get_task_status(self, task_id):
        """Handle task status request"""
        status_data = self.task_manager.get_task_status(task_id)
        if not status_data:
            return jsonify({
                'code': 2001,
                'data': {'message': 'Task not found'}
            }), 404
            
        # Update model URL if task is completed
        if status_data['status'] == 'success' and 'model' in status_data['output']:
            status_data['output']['model'] = status_data['output']['model']
            
        return jsonify({
            'code': 0,
            'data': status_data
        }), 200
        
    def run(self, host='0.0.0.0', port=5000):
        """Run the API server"""
        self.app.run(host=host, port=port)

if __name__ == '__main__':
    api = TrellisAPI()
    api.run()


