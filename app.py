import os
import json
import subprocess
import signal
import time
import base64
import logging
import shlex
import base64
import requests
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Paths (from container's perspective)
BASE_DIR = '/app'
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
MODELS_DIR = '/app/models'
OUTPUT_DIR = '/app/output'
STATIC_DIR = BASE_DIR
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# Model subdirectories
DIFFUSION_MODELS_DIR = os.path.join(MODELS_DIR, 'diffusion')
VAE_MODELS_DIR = os.path.join(MODELS_DIR, 'vae')
LLM_MODELS_DIR = os.path.join(MODELS_DIR, 'llm')
CLIP_MODELS_DIR = os.path.join(MODELS_DIR, 'clip')
T5_MODELS_DIR = os.path.join(MODELS_DIR, 't5')
LORA_MODELS_DIR = os.path.join(MODELS_DIR, 'loras')
EMBEDDINGS_DIR = os.path.join(MODELS_DIR, 'embeddings')

# SD.cpp binary paths
SD_SERVER_BINARY = '/usr/local/bin/sd-server'
SD_CLI_BINARY = '/usr/local/bin/sd-cli'

# Flask app
app = Flask(__name__,
static_folder=STATIC_DIR,
template_folder=TEMPLATES_DIR)

# Global state
sd_process = None
current_model = None
current_model_type = None
server_port = 8080

def get_file_size_human(size_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def list_model_files(directory):
    """List model files in a directory"""
    models = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.gguf', '.safetensors', '.ckpt', '.pt', '.bin']:
                    size = os.path.getsize(filepath)
                    models.append({
                        'name': filename,
                        'path': filepath,
                        'size': size,
                        'size_human': get_file_size_human(size)
                    })
    return sorted(models, key=lambda x: x['name'])

# Routes for static files
@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(STATIC_DIR, 'css'), filename)

@app.route('/scripts/<path:filename>')
def serve_scripts(filename):
    return send_from_directory(os.path.join(STATIC_DIR, 'scripts'), filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config')
def get_config():
    """Return the configuration file"""
    try:
        logger.debug(f"Loading config from: {CONFIG_PATH}")
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {CONFIG_PATH}")
        return jsonify({'error': 'Config file not found'}), 404
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/list_models')
def list_models():
    """List all available models by category"""
    return jsonify({
        'diffusion': list_model_files(DIFFUSION_MODELS_DIR),
        'vae': list_model_files(VAE_MODELS_DIR),
        'llm': list_model_files(LLM_MODELS_DIR),
        'clip': list_model_files(CLIP_MODELS_DIR),
        't5': list_model_files(T5_MODELS_DIR),
        'lora': list_model_files(LORA_MODELS_DIR),
        'embeddings': list_model_files(EMBEDDINGS_DIR)
    })

@app.route('/server_status')
def server_status():
    """Check if the SD server is running and responsive"""
    global sd_process, current_model, current_model_type

    process_running = sd_process is not None and sd_process.poll() is None
    server_responsive = False

    if process_running:
        try:
            import urllib.request
            req = urllib.request.urlopen(f'http://127.0.0.1:{server_port}', timeout=2)
            server_responsive = req.status == 200
        except:
            pass

    return jsonify({
        'process_running': process_running,
        'server_responsive': server_responsive,
        'current_model': current_model,
        'model_type': current_model_type,
        'port': server_port
    })

@app.route('/upload_init_image', methods=['POST'])
def upload_init_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    raw_data = file.read()
    
    try:
        from PIL import Image
        import io
        
        # Auto-detect + convert ANY format to PNG
        img = Image.open(io.BytesIO(raw_data))
        img = img.convert('RGBA')  # Ensure transparency support
        
        timestamp = int(datetime.now().timestamp())
        filename = f"{timestamp}.png"
        filepath = f"/tmp/{filename}"
        
        img.save(filepath, 'PNG')  # Force PNG output
        logger.info(f"Converted {file.filename} -> PNG {img.size}: {filepath}")
        
        return jsonify({'success': True, 'path': filepath})
        
    except Exception as e:
        logger.error(f"Image conversion failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a model and start the SD server"""
    global sd_process, current_model, current_model_type, server_port

    # Unload any existing model first
    if sd_process is not None:
        unload_model_internal()

    data = request.json
    model_type = data.get('model_type', 'flux')
    diffusion_model = data.get('diffusion_model')

    if not diffusion_model:
        return jsonify({'error': 'No diffusion model specified'}), 400

    # Build command using sd-server
    cmd = [SD_SERVER_BINARY,'--listen-ip', '0.0.0.0', '--listen-port', str(server_port)]
    cmd.extend(['--diffusion-model', diffusion_model])

    # Add optional models
    if data.get('vae'):
        cmd.extend(['--vae', data['vae']])
    if data.get('clip_l'):
        cmd.extend(['--clip_l', data['clip_l']])
    if data.get('clip_g'):
        cmd.extend(['--clip_g', data['clip_g']])
    if data.get('t5xxl'):
        cmd.extend(['--t5xxl', data['t5xxl']])
    if data.get('llm'):
        cmd.extend(['--llm', data['llm']])

    # Add options
    if data.get('vae_tiling'):
        cmd.append('--vae-tiling')
    if data.get('offload_to_cpu'):
        cmd.append('--offload-to-cpu')
    if data.get('diffusion_fa'):
        cmd.append('--diffusion-fa')
    if data.get('flow_shift'):
        cmd.extend(['--flow-shift', str(data['flow_shift'])])
    if data.get('lora_model_dir'):
        cmd.extend(['--lora-model-dir', data['lora_model_dir']])
    if data.get('embd_dir'):
        cmd.extend(['--embd-dir', data['embd_dir']])
    if data.get('threads'):
        cmd.extend(['--threads', str(data['threads'])])

    generation_params = {
        '--seed': data.get('seed', '-1.0'),
        '--cfg-scale': data.get('cfg_scale', '1.0'),
        '--guidance': data.get('guidance', '3.5'),
        '--sampling-method': data.get('sampling_method', 'euler'),
        '--scheduler': data.get('scheduler', 'discrete'),
        '--rng': data.get('rng', 'cuda'),
        '--sampler-rng':'cuda'
    }

    for flag, value in generation_params.items():
        if value is not None:
            cmd.extend([flag, str(value)])

    logger.info(f"Starting SD server with command: {' '.join(cmd)}")

    sd_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    try:
        current_model = os.path.basename(diffusion_model)
        current_model_type = model_type

        # Wait a bit for server to start
        time.sleep(3)

        if sd_process.poll() is not None:
            output = sd_process.stdout.read()
            return jsonify({'error': f'Server failed to start: {output}'}), 500

        return jsonify({
            'success': True,
            'message': 'Model loaded successfully',
            'model': current_model,
            'model_type': current_model_type
        })
    except Exception as e:
        logger.error(f"Failed to start SD server: {e}")
        return jsonify({'error': str(e)}), 500

def unload_model_internal():
    """Internal function to unload model"""
    global sd_process, current_model, current_model_type

    if sd_process is not None:
        sd_process.terminate()
        try:
            sd_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sd_process.kill()
        sd_process = None

    current_model = None
    current_model_type = None

@app.route('/unload_model', methods=['POST'])
def unload_model():
    """Unload the current model and stop the server"""
    unload_model_internal()
    return jsonify({'success': True, 'message': 'Model unloaded'})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400

    # Validate required parameters
    required_params = ['prompt', 'height', 'width', 'steps', 'cfg_scale', 'seed', 'sampler', 'scheduler', 'guidance']
    for param in required_params:
        if param not in data:
            return jsonify({'status': 'error', 'message': f'Missing required parameter: {param}'}), 400
    
    try:
        prompt = str(data['prompt'])
        negative_prompt = str(data.get('negative_prompt', ''))
        height = int(data['height'])
        width = int(data['width'])
        steps = int(data['steps'])
        cfg_scale = float(data['cfg_scale'])
        sampler = str(data['sampler'])
        scheduler = str(data['scheduler'])
        seed = int(data['seed'])
        guidance = float(data['guidance'])
        init_img = data.get('init_img')
        strength = float(data.get('strength', 0.75)) if data.get('strength') else None
    except (ValueError, TypeError) as e:
        return jsonify({'status': 'error', 'message': f'Invalid parameter type: {str(e)}'}), 400

    # Generate timestamp for output filename
    timestamp = int(datetime.now().timestamp())
    output_filename = f"output_{timestamp}.png"

    # Ensure output folder exists and get path
    try:
        output_folder = app.config.get('OUTPUT_FOLDER', '/app/output')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_filename)
    except Exception as e:
        app.logger.error(f'Failed to prepare output folder: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': f'Failed to prepare output folder: {str(e)}'
        }), 500

    # Generate the image
    success = generate_image(
        prompt, negative_prompt, height, width, steps,
        cfg_scale, seed, sampler, scheduler, guidance, output_path,
        init_img, strength
    )

    if success:
        return jsonify({
            'status': 'success',
            'output': output_filename,
            'message': 'Image generated successfully'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Image generation failed - check logs for details'
        }), 500

def generate_image(prompt, negative_prompt, height, width, steps, cfg_scale, seed, sampler, scheduler, guidance, output_path, init_img=None, strength=None):

    try:
        response = requests.get("http://localhost:5000/server_status", timeout=1)
        if response.status_code == 200:
            status = response.json()
            if status.get("process_running", False) and status.get("server_responsive", False):
                # Server is running, use API
                return generate_via_api(prompt, negative_prompt, height, width, steps, cfg_scale, seed, sampler, scheduler, guidance, output_path, init_img, strength)
    except:
        pass

    # Fall back to sd-cli
    return generate_via_cli(prompt, negative_prompt, height, width, steps, cfg_scale, seed, sampler, scheduler, guidance, output_path, init_img, strength)


def truncate_payload_values(payload_dict, max_value_length=100):
    """Truncate dict values for logging - input must be DICT"""
    import json
    payload_copy = json.loads(json.dumps(payload_dict))  # Deep copy
    
    def truncate_value(value):
        if isinstance(value, str) and len(value) > max_value_length:
            if any(prefix in value[:50] for prefix in ['/9j/4AAQ', 'iVBORw0KGgo', 'UklGR']):
                return f"{value[:30]}...[BASE64 {len(value)} chars]..."
            return f"{value[:max_value_length]}...[TRUNCATED {len(value)} chars]..."
        return value
    
    def truncate_recursive(obj):
        if isinstance(obj, dict):
            return {k: truncate_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [truncate_recursive(item) for item in obj]
        else:
            return truncate_value(obj)
    
    return truncate_recursive(payload_copy)


def truncate_response_text(response_text, max_value_length=100):
    """Truncate JSON response TEXT - input must be STR"""
    import json
    try:
        parsed = json.loads(response_text)
        def truncate_recursive(obj):
            if isinstance(obj, dict):
                return {k: truncate_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [truncate_recursive(item) for item in obj]
            else:
                return truncate_value(obj)
        
        def truncate_value(value):
            if isinstance(value, str) and len(value) > max_value_length:
                if any(prefix in str(value)[:50] for prefix in ['/9j/4AAQ', 'iVBORw0KGgo', 'UklGR']):
                    return f"{str(value)[:30]}...[BASE64 {len(str(value))} chars]..."
                return f"{str(value)[:max_value_length]}...[TRUNCATED]"
            return value
        
        truncated = truncate_recursive(parsed)
        return json.dumps(truncated, indent=2)
    except:
        return response_text[:500] + f"...[TEXT TRUNCATED {len(response_text)} chars]"


def generate_via_api(prompt, negative_prompt, height, width, steps, cfg_scale, seed, sampler, scheduler, guidance, output_path, init_img=None, strength=None):
    # Prepare the API request payload - using the exact format that works with curl
    payload = {
        "model": "sd-cpp-local",
        "prompt": prompt,
        "height": height,
        "width": width,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler": sampler,
        "scheduler": scheduler,
        "guidance": guidance
    }

    # Add init image if provided
    # CRITICAL: Base64 + denoise for sd-server img2img
    if init_img and strength:
        import base64
        with open(init_img, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        payload.update({
            "image": img_b64,        # Base64 data
            "denoise": strength,     # "denoise" NOT "strength"
            "init_image": img_b64    # Fallback field name
        })
    
    # Only include negative prompt if it's not empty
    if negative_prompt and negative_prompt.strip():
        payload["negative_prompt"] = negative_prompt

    try:
        log_payload = truncate_payload_values(payload)
        # Safe logging - shows structure without dumping megabytes
        app.logger.info(f"Sending API payload: {json.dumps(log_payload, indent=2)}")

        # Send request to the API
        response = requests.post(
            "http://localhost:8080/v1/images/generations",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=12000  # Increased timeout for larger images
        )
        app.logger.info(f"Full API response: {truncate_response_text(response.text )}")
        app.logger.info(f"API response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            app.logger.info(f"API response data keys: {list(data.keys())}")

            if data.get("data") and len(data["data"]) > 0:
                image_data = data["data"][0].get("b64_json")
                if not image_data:
                    app.logger.error("No b64_json in response data")
                    return False

                # Decode and save the base64 image
                import base64
                try:
                    decoded_data = base64.b64decode(image_data)

                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    with open(output_path, "wb") as f:
                        f.write(decoded_data)

                    # Verify file was written and has content
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        app.logger.info(f"Successfully wrote image to {output_path}")
                        app.logger.info(f"Image size: {os.path.getsize(output_path)} bytes")
                        return True
                    else:
                        app.logger.error(f"File was created but is empty: {output_path}")
                        return False
                except Exception as e:
                    app.logger.error(f"Failed to decode base64 image: {str(e)}")
                    return False
            else:
                app.logger.error("API response missing image data")
                return False
        else:
            app.logger.error(f"API request failed with status code: {response.status_code}")
            app.logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        app.logger.error(f"API request failed with error: {str(e)}", exc_info=True)
        return False

def generate_via_cli(prompt, negative_prompt, height, width, steps, cfg_scale, seed, sampler, scheduler, guidance, output_path, init_img=None, strength=None):
    # Original sd-cli implementation
    cmd = [
        "sd-cli", "generate",
        "--prompt", prompt,
        "--height", str(height),
        "--width", str(width),
        "--steps", str(steps),
        "--cfg-scale", str(cfg_scale),
        "--seed", str(seed),
        "--sampler", sampler,
        "--scheduler", scheduler,
        "--guidance", str(guidance),
        "--output", output_path
    ]

    # Add init image if provided
    if init_img:
        cmd.extend(["--init-img", init_img])
        if strength is not None:
            cmd.extend(["--strength", str(strength)])

    if negative_prompt:
        cmd.extend(["--negative-prompt", negative_prompt])

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"sd-cli generation failed: {str(e)}")
        return False

@app.route('/generate_cli', methods=['POST'])
def generate_cli():
    """Generate an image using CLI mode (no server)"""
    data = request.json
    prompt = data.get('prompt', '')
    model_args = data.get('model_args', {})

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    diffusion_model = model_args.get('diffusion_model')
    if not diffusion_model:
        return jsonify({'error': 'No diffusion model specified'}), 400

    # Generate output filename
    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f'output_{timestamp}.png')

    # Build command
    cmd = [SD_CLI_BINARY, '--mode', 'img_gen']
    cmd.extend(['--diffusion-model', diffusion_model])
    cmd.extend(['--prompt', prompt])
    cmd.extend(['--output', output_path])
    cmd.extend(['--width', str(data.get('width', 512))])
    cmd.extend(['--height', str(data.get('height', 512))])
    cmd.extend(['--steps', str(data.get('steps', 20))])
    cmd.extend(['--cfg-scale', str(data.get('cfg_scale', 7.0))])
    cmd.extend(['--seed', str(data.get('seed',-1))])
    cmd.extend(['--sampling-method', data.get('sampler', 'euler')])
    cmd.extend(['--scheduler', data.get('scheduler', 'discrete')])

    if data.get('negative_prompt'):
        cmd.extend(['--negative-prompt', data['negative_prompt']])

    if data.get('guidance'):
        cmd.extend(['--guidance', str(data['guidance'])])

    # Add init image if provided
    if data.get('init_img'):
        cmd.extend(['--init-img', data['init_img']])
        if data.get('strength'):
            cmd.extend(['--strength', str(data['strength'])])

    # Add optional models
    if model_args.get('vae'):
        cmd.extend(['--vae', model_args['vae']])
    if model_args.get('clip_l'):
        cmd.extend(['--clip_l', model_args['clip_l']])
    if model_args.get('clip_g'):
        cmd.extend(['--clip_g', model_args['clip_g']])
    if model_args.get('t5xxl'):
        cmd.extend(['--t5xxl', model_args['t5xxl']])
    if model_args.get('llm'):
        cmd.extend(['--llm', model_args['llm']])

    # Add options
    if model_args.get('vae_tiling'):
        cmd.append('--vae-tiling')
    if model_args.get('diffusion_fa'):
        cmd.append('--diffusion-fa')

    logger.info(f"Running CLI generation: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60000)

        if result.returncode != 0:
            return jsonify({'error': f'Generation failed: {result.stderr}'}), 500

        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({'image': image_data, 'path': output_path})
        else:
            return jsonify({'error': 'Output image not found'}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Generation timed out'}), 500
    except Exception as e:
        logger.error(f"CLI generation failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/list_outputs')
def list_outputs():
    """List generated images and videos"""
    files = []
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            filepath = os.path.join(OUTPUT_DIR, filename)
            ext = filename.lower().split('.')[-1]

            # Check for both images and videos
            if ext in ['png', 'jpg', 'jpeg', 'webp', 'mp4', 'avi', 'mov', 'webm']:
                file_type = 'video' if ext in ['mp4', 'avi', 'mov', 'webm'] else 'image'
                files.append({
                    'name': filename,
                    'url': f'/output/{filename}',
                    'path': filepath,
                    'mtime': os.path.getmtime(filepath),
                    'type': file_type,
                    'size': os.path.getsize(filepath),
                    'size_human': get_file_size_human(os.path.getsize(filepath))
                })

    # Sort by modification time, newest first
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return jsonify({'files': files})

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve generated images"""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/server_log')
def server_log():
    """Get server log output"""
    global sd_process

    if sd_process is None:
        return jsonify({'log': 'No server running'})

    # This is a simplified version - in practice you'd want to capture logs properly
    return jsonify({'log': 'Server is running. Check container logs for details.'})

@app.route('/generate_video', methods=['POST'])
def generate_video():
    """Generate a video using the loaded model (server mode)"""
    global server_port

    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        import urllib.request

        # Build payload for sd-server /generate endpoint
        payload = {
            'prompt': prompt,
            'height': data.get('height', 480),
            'width': data.get('width', 832),
            'steps': data.get('steps', 30),
            'cfg_scale': data.get('cfg_scale', 5.0),
            'seed': data.get('seed',-1),
            'sampler': data.get('sampler', 'euler'),
            'scheduler': data.get('scheduler', 'sgm_uniform'),
            'video_frames': data.get('video_frames', 33),
            'fps': data.get('fps', 24)
        }

        # Add negative prompt if provided
        if data.get('negative_prompt'):
            payload['negative_prompt'] = data.get('negative_prompt')

        # Add guidance if provided
        if data.get('guidance'):
            payload['guidance'] = data.get('guidance')

        # Add moe_boundary if provided
        if data.get('moe_boundary'):
            payload['moe_boundary'] = data.get('moe_boundary')

        # Add init image if provided
        if data.get('init_image'):
            payload['init_img'] = data.get('init_image')

        # Generate output filename
        timestamp = int(time.time())
        output_path = os.path.join(OUTPUT_DIR, f'output_video_{timestamp}.mp4')
        payload['output'] = output_path

        payload_json = json.dumps(payload).encode('utf-8')

        logger.info(f"Sending video request to server: {payload}")

        req = urllib.request.Request(
            f'http://127.0.0.1:{server_port}/generate',
            data=payload_json,
            headers={'Content-Type': 'application/json'}
        )

        response = urllib.request.urlopen(req, timeout=600000)
        result = json.loads(response.read().decode('utf-8'))

        # Check if output file was created
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({'video': video_data, 'path': output_path})
        elif 'video' in result:
            # If the server returns base64 video directly
            return jsonify({'video': result['video']})

        return jsonify(result)
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_video_cli', methods=['POST'])
def generate_video_cli():
    """Generate a video using CLI mode (no server)"""
    data = request.json
    prompt = data.get('prompt', '')
    model_args = data.get('model_args', {})

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    diffusion_model = model_args.get('diffusion_model')
    if not diffusion_model:
        return jsonify({'error': 'No diffusion model specified'}), 400

    # Generate output filename
    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f'output_video_{timestamp}')


    # Build command - use vid_gen mode
    cmd = [SD_CLI_BINARY, '--mode', 'vid_gen']
    cmd.extend(['--diffusion-model', diffusion_model])
    cmd.extend(['--prompt', prompt])
    cmd.extend(['--output', output_path])
    cmd.extend(['--width', str(data.get('width', 832))])
    cmd.extend(['--height', str(data.get('height', 480))])
    cmd.extend(['--steps', str(data.get('steps', 30))])
    cmd.extend(['--cfg-scale', str(data.get('cfg_scale', 5.0))])
    cmd.extend(['--seed', str(data.get('seed'))])
    cmd.extend(['--sampling-method', data.get('sampler', 'euler')])
    cmd.extend(['--scheduler', data.get('scheduler', 'sgm_uniform')])
    cmd.extend(['--video-frames', str(data.get('video_frames', 33))])

    if data.get('negative_prompt'):
        cmd.extend(['--negative-prompt', data['negative_prompt']])

    if data.get('guidance'):
        cmd.extend(['--guidance', str(data['guidance'])])

    flow_shift = data.get('flow_shift') or (model_args.get('flow_shift') if model_args else None)
    if flow_shift:
        cmd.extend(['--flow-shift', str(flow_shift)])

    # Add optional models
    if model_args.get('vae'):
        cmd.extend(['--vae', model_args['vae']])
    if model_args.get('t5xxl'):
        cmd.extend(['--t5xxl', model_args['t5xxl']])

    # Add options
    if model_args.get('vae_tiling'):
        cmd.append('--vae-tiling')
    if model_args.get('diffusion_fa'):
        cmd.append('--diffusion-fa')

    logger.info(f"Running CLI video generation: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60000)

        if result.returncode != 0:
            return jsonify({'error': f'Video generation failed: {result.stderr}'}), 500

        # Look for the output video file
        video_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f'output_video_{timestamp}')]

        if video_files:
            video_path = os.path.join(OUTPUT_DIR, video_files[0])
            with open(video_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({'video': video_data, 'path': video_path})
        else:
            return jsonify({'error': 'Output video not found'}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Video generation timed out'}), 500
    except Exception as e:
        logger.error(f"CLI video generation failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
