import os
import json
import subprocess
import signal
import time
import base64
import logging
import shlex
import requests
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)  # Changed from DEBUG to INFO
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
    logger.info(f"Load model request: {json.dumps(truncate_payload_values(data), indent=2)}")
    
    model_type = data.get('model_type', 'flux')
    diffusion_model = data.get('diffusion_model')

    if not diffusion_model:
        return jsonify({'error': 'No diffusion model specified'}), 400

    # Build command using sd-server
    cmd = [SD_SERVER_BINARY,'--listen-ip', '0.0.0.0', '--listen-port', str(server_port)]
    if model_type in ['SD3', 'SD3.5']:
        cmd.extend(['--model', diffusion_model])
    else:
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

def is_server_running():
    """Check if the SD server is running and responsive"""
    try:
        response = requests.get(f"http://localhost:{server_port}", timeout=1)
        return response.status_code == 200
    except:
        return False

@app.route('/generate', methods=['POST'])
def generate():
    """Generate image - intelligently routes to API or CLI"""
    data = request.json
    logger.info(f"Generate request: {json.dumps(truncate_payload_values(data), indent=2)}")
    
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
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
    except Exception as e:
        logger.error(f'Failed to prepare output folder: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': f'Failed to prepare output folder: {str(e)}'
        }), 500

    # Check if server is running and route accordingly
    if is_server_running():
        logger.info("Routing to API (server is running)")
        success = generate_via_api(
            prompt, negative_prompt, height, width, steps,
            cfg_scale, seed, sampler, scheduler, guidance, output_path,
            init_img, strength
        )
    else:
        # Fall back to CLI mode - need model_args from request
        logger.info("Routing to CLI (server is not running)")
        model_args = data.get('model_args', {})
        logger.info(f"CLI model_args: {json.dumps(truncate_payload_values(model_args), indent=2)}")
        success = generate_via_cli(
            prompt, negative_prompt, height, width, steps,
            cfg_scale, seed, sampler, scheduler, guidance, output_path,
            init_img, strength, model_args
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


def generate_via_api(prompt, negative_prompt, height, width, steps, cfg_scale, seed, sampler, scheduler, guidance, output_path, init_img=None, strength=None):
    """Generate image using the sd-server API"""
    if init_img:
        # Image-to-image: use /v1/images/edits with multipart/form-data
        url = f"http://localhost:{server_port}/v1/images/edits"

        # Read image file into memory as raw bytes
        with open(init_img, 'rb') as f:
            image_bytes = f.read()

        # Build sd_cpp_extra_args JSON - THIS is what the server actually reads
        extra_args = {
            "width": width,              # ← Include dimensions here too
            "height": height,            # ← Include dimensions here too
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,
            "seed": seed,
            "strength": strength if strength is not None else 0.75
        }
        
        if negative_prompt and negative_prompt.strip():
            extra_args["negative_prompt"] = negative_prompt
        
        # Embed extra args in the prompt using the expected format
        prompt_with_args = f"{prompt}<sd_cpp_extra_args>{json.dumps(extra_args)}</sd_cpp_extra_args>"

        # Minimal payload - server ignores most form fields anyway
        payload_data = {
            'model': 'sd-cpp-local',
            'prompt': prompt_with_args,  # ← Contains all the actual parameters
            'size': f'{width}x{height}',  # ← Server does read this
            'n': '1'
        }

        try:
            log_payload = {k: v for k, v in payload_data.items()}
            log_payload['prompt'] = prompt  # Log without the extra args for readability
            logger.info(f"Sending Multipart payload to {url}: {json.dumps(log_payload, indent=2)}")
            logger.info(f"Extra args (embedded in prompt): {json.dumps(extra_args, indent=2)}")

            response = requests.post(
                url,
                files={'image[]': ('init_image.png', image_bytes, 'image/png')},
                data=payload_data,
                timeout=120000
            )
            
            logger.info(f"API response status: {response.status_code}")
            
            # Debug: Log response to catch HTML errors
            if response.status_code != 200:
                logger.error(f"Response text (first 500 chars): {response.text[:500]}")
            
            logger.debug(f"Full API response: {truncate_response_text(response.text)}")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"API response data keys: {list(data.keys())}")
                logger.debug(f"API response data (truncated): {json.dumps(truncate_payload_values(data), indent=2)}")

                if data.get("data") and len(data["data"]) > 0:
                    image_data = data["data"][0].get("b64_json")
                    if not image_data:
                        logger.error("No b64_json in response data")
                        return False

                    # Decode and save the base64 image
                    try:
                        import base64 as b64
                        decoded_data = b64.b64decode(image_data)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                        with open(output_path, "wb") as f:
                            f.write(decoded_data)

                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            logger.info(f"Successfully wrote image to {output_path}")
                            logger.info(f"Image size: {os.path.getsize(output_path)} bytes")
                            return True
                        else:
                            logger.error(f"File was created but is empty: {output_path}")
                            return False
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {str(e)}")
                        return False
                else:
                    logger.error("API response missing image data")
                    return False
            else:
                logger.error(f"API request failed with status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False

        except Exception as e:
            logger.error(f"API request failed with error: {str(e)}", exc_info=True)
            return False

    else:
        # Text-to-image: use /v1/images/generations with JSON
        url = f"http://localhost:{server_port}/v1/images/generations"
        
        # For txt2img, also use sd_cpp_extra_args for consistency
        extra_args = {
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,
            "seed": seed
        }
        
        if negative_prompt and negative_prompt.strip():
            extra_args["negative_prompt"] = negative_prompt
        
        prompt_with_args = f"{prompt}<sd_cpp_extra_args>{json.dumps(extra_args)}</sd_cpp_extra_args>"
        
        payload = {
            "model": "sd-cpp-local",
            "prompt": prompt_with_args,
            "size": f'{width}x{height}',
            "n": 1
        }

        try:
            log_payload = truncate_payload_values(payload.copy())
            log_payload['prompt'] = prompt
            logger.info(f"Sending API payload to {url}: {json.dumps(log_payload, indent=2)}")
            logger.info(f"Extra args (embedded in prompt): {json.dumps(extra_args, indent=2)}")

            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=12000
            )
            logger.info(f"API response status: {response.status_code}")
            logger.debug(f"Full API response: {truncate_response_text(response.text)}")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"API response data keys: {list(data.keys())}")
                logger.debug(f"API response data (truncated): {json.dumps(truncate_payload_values(data), indent=2)}")

                if data.get("data") and len(data["data"]) > 0:
                    image_data = data["data"][0].get("b64_json")
                    if not image_data:
                        logger.error("No b64_json in response data")
                        return False

                    # Decode and save the base64 image
                    try:
                        import base64 as b64
                        decoded_data = b64.b64decode(image_data)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                        with open(output_path, "wb") as f:
                            f.write(decoded_data)

                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            logger.info(f"Successfully wrote image to {output_path}")
                            logger.info(f"Image size: {os.path.getsize(output_path)} bytes")
                            return True
                        else:
                            logger.error(f"File was created but is empty: {output_path}")
                            return False
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {str(e)}")
                        return False
                else:
                    logger.error("API response missing image data")
                    return False
            else:
                logger.error(f"API request failed with status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"API request failed with error: {str(e)}", exc_info=True)
            return False


def generate_via_cli(prompt, negative_prompt, height, width, steps, cfg_scale, seed, sampler, scheduler, guidance, output_path, init_img=None, strength=None, model_args=None):
    """Generate image using sd-cli"""
    if model_args is None:
        logger.error("CLI mode requires model_args")
        return False

    logger.info(f"CLI generation params: prompt={prompt[:50]}..., size={width}x{height}, steps={steps}")
    
    diffusion_model = model_args.get('diffusion_model')
    if not diffusion_model:
        logger.error("No diffusion model specified for CLI mode")
        return False

    cmd = [SD_CLI_BINARY, '--mode', 'img_gen']
    cmd.extend(['--diffusion-model', diffusion_model])
    cmd.extend(['--prompt', prompt])
    cmd.extend(['--output', output_path])
    cmd.extend(['--width', str(width)])
    cmd.extend(['--height', str(height)])
    cmd.extend(['--steps', str(steps)])
    cmd.extend(['--cfg-scale', str(cfg_scale)])
    cmd.extend(['--seed', str(seed)])
    cmd.extend(['--sampling-method', sampler])
    cmd.extend(['--scheduler', scheduler])
    cmd.extend(['--guidance', str(guidance)])

    if negative_prompt:
        cmd.extend(['--negative-prompt', negative_prompt])

    # Add init image if provided
    if init_img:
        cmd.extend(['--init-img', init_img])
        if strength is not None:
            cmd.extend(['--strength', str(strength)])

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

    logger.info(f"Running CLI generation: {' '.join(cmd[:5])}... (truncated)")
    logger.debug(f"Full CLI command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60000)

        if result.returncode != 0:
            logger.error(f"CLI generation failed: {result.stderr}")
            return False

        if os.path.exists(output_path):
            logger.info(f"Successfully generated image via CLI: {output_path}")
            return True
        else:
            logger.error("Output image not found after CLI generation")
            return False
    except subprocess.TimeoutExpired:
        logger.error("CLI generation timed out")
        return False
    except Exception as e:
        logger.error(f"CLI generation failed: {e}")
        return False

@app.route('/generate_video', methods=['POST'])
def generate_video():
    """Generate a video using the loaded model (server mode)"""
    global server_port

    data = request.json
    logger.info(f"Generate video request: {json.dumps(truncate_payload_values(data), indent=2)}")
    
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

        logger.info(f"Sending video request to server: {json.dumps(truncate_payload_values(payload), indent=2)}")

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
