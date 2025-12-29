# gallery.py
import os
from flask import Blueprint, render_template, send_from_directory
from datetime import datetime

gallery_bp = Blueprint('gallery', __name__)

@gallery_bp.route('/gallery')
def gallery():
    # Get the output directory path (same as where generated images are saved)
    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    # Get all image files sorted by creation date (newest first)
    try:
        files = []
        for filename in os.listdir(output_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                filepath = os.path.join(output_dir, filename)
                stat = os.stat(filepath)
                files.append({
                    'name': filename,
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime),
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })

        # Sort by creation date (newest first)
        files.sort(key=lambda x: x['created'], reverse=True)

    except Exception as e:
        files = []
        print(f"Error reading gallery files: {e}")

    return render_template('gallery.html', images=files)

@gallery_bp.route('/gallery/<filename>')
def gallery_image(filename):
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    return send_from_directory(output_dir, filename)
