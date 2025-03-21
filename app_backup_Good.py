from flask import Flask, request, render_template, jsonify, send_file
import yt_dlp
import os
from datetime import datetime
import uuid

app = Flask(__name__)

# Create a directory for storing subtitle files
SUBTITLE_DIR = "static/subtitles"
os.makedirs(SUBTITLE_DIR, exist_ok=True)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/download_subtitle', methods=['POST'])
def download_subtitle():
    """API endpoint to download YouTube subtitles"""
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    
    if not youtube_url:
        return jsonify({'status': 'error', 'message': 'No YouTube URL provided'}), 400
    
    # Generate a unique identifier for this download
    unique_id = str(uuid.uuid4())[:8]
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_prefix = f"video_{date_string}_{unique_id}_"
    
    try:
        with yt_dlp.YoutubeDL({'skip_download': True}) as ydl:
            # Extract video information
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'video')
            subtitles = info.get('subtitles', {})  # Manually provided subtitles
            auto_subtitles = info.get('automatic_captions', {})  # Auto-generated subtitles
            
            # Determine whether to use manual or auto-generated subtitles
            use_auto_subtitles = False
            if subtitles:
                print(f"Found manually provided subtitles for '{video_title}'.")
            elif auto_subtitles:
                print(f"No manual subtitles found for '{video_title}', using auto-generated subtitles.")
                subtitles = auto_subtitles
                use_auto_subtitles = True
            else:
                return jsonify({
                    'status': 'error',
                    'message': f"Video '{video_title}' does not have any subtitles (manual or auto-generated)."
                }), 404
            
            # Configure yt-dlp options to download the selected subtitles
            ydl_opts = {
                'skip_download': True,  # Do not download the video
                'writesubtitles': True,  # Download subtitles
                'writeautomaticsub': use_auto_subtitles,  # Download auto-generated subtitles if needed
                'subtitlesformat': 'vtt',  # Subtitle format
                'outtmpl': os.path.join(SUBTITLE_DIR, f'{output_prefix}%(title)s.%(ext)s'),  # Output filename template
            }
            
            # Download subtitles
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([youtube_url])
            
            # Collect information about downloaded subtitle files
            subtitle_files = []
            for lang, sub_info in subtitles.items():
                subtitle_file = f"{output_prefix}{video_title}.{lang}.vtt"
                full_path = os.path.join(SUBTITLE_DIR, subtitle_file)
                
                if os.path.exists(full_path):
                    subtitle_files.append({
                        'language': lang,
                        'filename': subtitle_file,
                        'path': f'/static/subtitles/{subtitle_file}',
                        'type': 'auto-generated' if use_auto_subtitles else 'manual'
                    })
            
            if subtitle_files:
                return jsonify({
                    'status': 'success',
                    'message': f"Subtitles for '{video_title}' downloaded successfully",
                    'video_title': video_title,
                    'subtitles': subtitle_files
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': "Subtitle download failed."
                }), 500
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
