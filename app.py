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
import requests
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
# Try to load from .env if exists, but don't fail if it doesn't
if os.path.exists('.env'):
    load_dotenv()

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
        self.storage_input_dir = config["storage"]['prefix'] + "input_images"
        self.storage_output_dir = config["storage"]['prefix'] + "output_tasks"
        
        # Create necessary directories
        os.makedirs(self.IMAGES_DIR, exist_ok=True)
        os.makedirs(self.TASKS_DIR, exist_ok=True)
        
        # Initialize storage
        self.storage = get_storage_provider(config["storage"])
        self.storage_type = config["storage"]['provider']
        
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
        
    def process_image(self, image: Image.Image, params: dict, progress_callback=None) -> dict:
        """
        Process an image using Trellis pipeline with dynamic progress tracking.

        Args:
            image (Image.Image): The input image.
            params (dict): Configuration parameters for the pipeline.
            progress_callback (function): A callback to report progress.

        Returns:
            dict: The output of the pipeline.
        """
        # Run the pipeline with progress tracking callbacks
        outputs = self.pipeline.run_with_progress(
            image=image,
            seed=params.get('geometry_seed', 42),
            formats=["gaussian", "mesh"],
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": params.get('sparse_structure_steps', 20),
                "cfg_strength": params.get('sparse_structure_strength', 7.5),
            },
            slat_sampler_params={
                "steps": params.get('slat_steps', 20),
                "cfg_strength": params.get('slat_strength', 3.0),
            },
            progress_callback=progress_callback
        )

        return outputs

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
        """Save task metadata to task directory and storage provider"""
        task_dir = self.get_task_dir(task_id)
        metadata_path = os.path.join(task_dir, 'metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        storage_path = f"{self.config.storage_output_dir}/{task_id}/metadata.json"
        return self.config.storage.upload_file(metadata_path, storage_path)
        
    def process_task(self, task_id: str, image_path: str, params: dict):
        """Process a task in background"""
        try:
            # Phase 1: Initial setup (0-2%)
            start_time = time.time()
            logger.info(f"Starting task processing for task_id: {task_id}")
            self.tasks[task_id]['status'] = 'processing'
            self.tasks[task_id]['progress'] = 0
            
            # Copy input image
            task_dir = self.get_task_dir(task_id)
            task_image_path = os.path.join(task_dir, 'input.png')
            shutil.copy2(image_path, task_image_path)
            logger.debug(f"Task {task_id}: Copied input image, progress: 1%")
            self.tasks[task_id]['progress'] = 1
            
            # Upload input image to storage provider
            storage_input_path = f"{self.config.storage_output_dir}/{task_id}/input.png"
            self.config.storage.upload_file(task_image_path, storage_input_path)
            logger.debug(f"Task {task_id}: Uploaded input image, progress: 2%")
            self.tasks[task_id]['progress'] = 2
            
            # Save and upload parameters
            params_path = os.path.join(task_dir, 'params.json')
            with open(params_path, 'w') as f:
                json.dump(params, f)
            storage_params_path = f"{self.config.storage_output_dir}/{task_id}/params.json"
            self.config.storage.upload_file(params_path, storage_params_path)
            logger.debug(f"Task {task_id}: Uploaded parameters, progress: 3%")
            self.tasks[task_id]['progress'] = 3
            
            setup_time = time.time() - start_time
            logger.debug(f"Phase 1 (Setup) took {setup_time:.2f}s")
            
            # Phase 2: Model Processing (2-35%)
            model_start = time.time()
            logger.debug(f"Task {task_id}: Starting Phase 2 - Model Processing")
            def model_progress_callback(progress):
                """Maps model progress (0-100) to Phase 2 range (2-35)"""
                overall_progress = 2 + (progress * 0.33)  # Map 0-100 to 2-35
                self.tasks[task_id]['progress'] = overall_progress
                logger.debug(f"Task {task_id}: Model processing progress: {overall_progress:.2f}%")
            
            image = Image.open(task_image_path)
            outputs = self.model.process_image(image, params, progress_callback=model_progress_callback)
            model_time = time.time() - model_start
            logger.debug(f"Phase 2 (Model Processing) took {model_time:.2f}s")
            
            # Phase 3: Post-processing (35-100%)
            post_start = time.time()
            logger.debug(f"Task {task_id}: Starting Post-processing")

            def postprocessing_progress_callback(progress):
                """Maps postprocessing progress (0-100) to Phase 3 range (35-90)"""
                overall_progress = 35 + (progress * 0.55)  # Map 0-100 to 35-90
                self.tasks[task_id]['progress'] = overall_progress
                logger.debug(f"Task {task_id}: Post-processing progress: {overall_progress:.2f}%")

            # Generate GLB
            glb_start = time.time()
            logger.debug(f"Task {task_id}: Starting GLB generation")
            self.tasks[task_id]['progress'] = 35
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.95,
                texture_size=1024,
                verbose=False,
                use_vertex_colors=False,
                progress_callback=postprocessing_progress_callback
            )
            glb_time = time.time() - glb_start
            logger.debug(f"GLB generation took {glb_time:.2f}s")
            self.tasks[task_id]['progress'] = 90
            
            # Save and upload results
            save_start = time.time()
            glb_path = os.path.join(task_dir, 'model.glb')
            glb.export(glb_path)
            save_time = time.time() - save_start
            logger.debug(f"GLB save took {save_time:.2f}s")
            self.tasks[task_id]['progress'] = 95
            
            upload_start = time.time()
            storage_model_path = f"{self.config.storage_output_dir}/{task_id}/model.glb"
            download_temp_url = self.config.storage.upload_file(glb_path, storage_model_path)
            upload_time = time.time() - upload_start
            logger.debug(f"Storage provider upload took {upload_time:.2f}s")
            
            post_time = time.time() - post_start
            logger.debug(f"Phase 3 (Post-processing) took {post_time:.2f}s")
            
            total_time = time.time() - start_time
            logger.info(f"""Task timing breakdown:
                Setup: {setup_time:.2f}s ({(setup_time/total_time)*100:.1f}%)
                Model: {model_time:.2f}s ({(model_time/total_time)*100:.1f}%)
                Post-processing: {post_time:.2f}s ({(post_time/total_time)*100:.1f}%)
                  - GLB Generation: {glb_time:.2f}s
                  - Save: {save_time:.2f}s
                  - Upload: {upload_time:.2f}s
                Total: {total_time:.2f}s
            """)
            
            # Complete task
            if self.config.storage_type == "s3":
                s3_url = f"s3://{self.config.storage.bucket_name}/{storage_model_path}"
            else:
                s3_url = ""
            if self.config.storage_type == "gcs":
                gcs_url = f"https://storage.googleapis.com/{self.config.storage.bucket_name}/{storage_model_path}"  
            else:
                gcs_url = ""
            self.tasks[task_id]['status'] = 'completed'
            self.tasks[task_id]['progress'] = 100
            self.tasks[task_id]['output'] = {'model': download_temp_url, "s3_url": s3_url, "gcs_url": gcs_url}
            
            self.save_task_metadata(task_id, self.tasks[task_id])
            
            # Cleanup
            shutil.rmtree(task_dir)
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
            self.tasks[task_id]['status'] = 'failed'
            self.tasks[task_id]['error'] = str(e)
            self.save_task_metadata(task_id, self.tasks[task_id])
            if os.path.exists(task_dir):
                shutil.rmtree(task_dir)
                logger.debug(f"Task {task_id}: Cleaned up task directory after failure")
                
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
        
    def is_url_expired(self, url: str) -> bool:
        """Check if a URL is expired by making a HEAD request"""
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            return response.status_code != 200
        except requests.RequestException:
            return True

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

        # generate a new presigned url for the model if the task is completed and the model url is expired
        if status_data['status'] == 'completed':
            try:
                model_url = status_data["output"].get("model")
                
                # Check if we need to generate a new URL
                need_new_url = (
                    not model_url or 
                    (isinstance(model_url, str) and self.is_url_expired(model_url))
                )
                if need_new_url:
                    storage_url = f"{self.config.storage_output_dir}/{task_id}/model.glb"
                    status_data['output']['model'] = self.config.storage.get_url(storage_url)
                    logger.debug(f"Generated new presigned URL for task {task_id}")
            except Exception as e:
                logger.error(f"Failed to generate presigned URL for task {task_id}: {str(e)}")
                # Keep the existing URL if we fail to generate a new one
        
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
            
            storage_path = f"{self.config.storage_input_dir}/{image_token}.png"
            storage_url = self.config.storage.upload_file(temp_path, storage_path)
            logger.info(f"Uploaded file to storage provider: {storage_path}")
            
            os.remove(temp_path)
            logger.debug(f"Cleaned up temporary file: {temp_path}")
            
            return jsonify({
                'code': 0,
                'data': {
                    'message': 'Image uploaded successfully',
                    'image_token': image_token,
                    'storage_url': storage_url,
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
                image_path_in_storage_url = f"{self.config.storage_input_dir}/{file_token}.png"
                download_path = os.path.join(self.config.IMAGES_DIR, f"{file_token}.png")
                self.config.storage.download_file(image_path_in_storage_url, download_path)
                
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


