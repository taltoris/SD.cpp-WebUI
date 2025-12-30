import os
import json
import subprocess
import signal
import time
import base64
import logging
import shlex
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

# SD.cpp binary path
SD_BINARY = '/usr/local/bin/sd-cli'

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
            req = urllib.request.urlopen(f'http://127.0.0.1:{server_port}/health', timeout=2)
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
    
    # Build command
    cmd = [SD_BINARY, '--mode', 'server', '--port', str(server_port)]
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
        cmd.append('--diffusion-fa')
    if data.get('diffusion_fa'):
        cmd.append('--diffusion-fa')
    if data.get('lora_model_dir'):
        cmd.extend(['--lora-model-dir', data['lora_model_dir']])
    if data.get('embd_dir'):
        cmd.extend(['--embd-dir', data['embd_dir']])
    if data.get('threads'):
        cmd.extend(['--threads', str(data['threads'])])
    
    logger.info(f"Starting SD server with command: {' '.join(cmd)}")
    
    try:
        sd_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        current_model = os.path.basename(diffusion_model)
        current_model_type = model_type
        
        # Wait a bit for server to start
        time.sleep(2)
        
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
    """Generate an image using the loaded model (server mode)"""
    global server_port
    
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        import urllib.request
        import urllib.parse
        
        payload = json.dumps({
            'prompt': prompt,
            'negative_prompt': data.get('negative_prompt', ''),
            'width': data.get('width', 512),
            'height': data.get('height', 512),
            'steps': data.get('steps', 20),
            'cfg_scale': data.get('cfg_scale', 1.0),
            'seed': data.get('seed', -1),
            'sampler': data.get('sampler', 'euler'),
            'scheduler': data.get('scheduler', 'sgm_uniform')
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f'http://127.0.0.1:{server_port}/txt2img',
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        response = urllib.request.urlopen(req, timeout=300)
        result = json.loads(response.read().decode('utf-8'))
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({'error': str(e)}), 500

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
    cmd = [SD_BINARY, '--mode', 'img_gen']
    cmd.extend(['--diffusion-model', diffusion_model])
    cmd.extend(['--prompt', shlex.quote(prompt)])
    cmd.extend(['--output', output_path])
    cmd.extend(['--width', str(data.get('width', 512))])
    cmd.extend(['--height', str(data.get('height', 512))])
    cmd.extend(['--steps', str(data.get('steps', 20))])
    cmd.extend(['--cfg-scale', str(data.get('cfg_scale', 7.0))])
    cmd.extend(['--seed', str(data.get('seed', -1))])
    cmd.extend(['--sampling-method', data.get('sampler', 'euler')])
    cmd.extend(['--scheduler', data.get('scheduler', 'normal')])
    
    if data.get('negative_prompt'):
        cmd.extend(['--negative-prompt', shlex.quote(data['negative_prompt'])])
    # Add guidance if provided
    if data.get('guidance'):
        cmd.extend(['--guidance', str(data['guidance'])])

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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
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
        import urllib.parse

        payload = json.dumps({
            'prompt': prompt,
            'negative_prompt': data.get('negative_prompt', ''),
            'width': data.get('width', 832),
            'height': data.get('height', 480),
            'steps': data.get('steps', 30),
            'cfg_scale': data.get('cfg_scale', 5.0),
            'seed': data.get('seed', -1),
            'sampler': data.get('sampler', 'euler'),
            'scheduler': data.get('scheduler', 'sgm_uniform'),
            'guidance': data.get('guidance', 3.5),
            'video_frames': data.get('video_frames', 33),
            'fps': data.get('fps', 24)
        }).encode('utf-8')

        req = urllib.request.Request(
            f'http://127.0.0.1:{server_port}/vid_gen',
            data=payload,
            headers={'Content-Type': 'application/json'}
        )

        response = urllib.request.urlopen(req, timeout=600)
        result = json.loads(response.read().decode('utf-8'))

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
    cmd = [SD_BINARY, '--mode', 'vid_gen']
    cmd.extend(['--diffusion-model', diffusion_model])
    cmd.extend(['--prompt', prompt])
    cmd.extend(['--output', output_path])
    cmd.extend(['--width', str(data.get('width', 832))])
    cmd.extend(['--height', str(data.get('height', 480))])
    cmd.extend(['--steps', str(data.get('steps', 30))])
    cmd.extend(['--cfg-scale', str(data.get('cfg_scale', 5.0))])
    cmd.extend(['--seed', str(data.get('seed', -1))])
    cmd.extend(['--sampling-method', data.get('sampler', 'euler')])
    cmd.extend(['--scheduler', data.get('scheduler', 'sgm_uniform')])
    cmd.extend(['--video-frames', str(data.get('video_frames', 33))])

    if data.get('negative_prompt'):
        cmd.extend(['--negative-prompt', data['negative_prompt']])
    
    # Add guidance if provided
    if data.get('guidance'):
        cmd.extend(['--guidance', str(data['guidance'])])

    # Add flow-shift if provided
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

        if result.returncode != 0:
            return jsonify({'error': f'Video generation failed: {result.stderr}'}), 500

        # Look for the output video file (sd-cli typically adds .mp4 extension)
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
