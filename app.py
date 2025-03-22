from flask import Flask, request, jsonify, render_template
import subprocess
import os
import requests
from bs4 import BeautifulSoup
import whisper
from pydub import AudioSegment
import logging
import json
import shutil
import time

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Available Whisper models
WHISPER_MODELS = {}
SELECTED_MODEL = "small"  # Default to small, change to "medium" for better accuracy

# Audio segment length (in milliseconds, 15 minutes per segment)
SEGMENT_LENGTH_MS = 15 * 60 * 1000  # 15 minutes

# Temporary directory for audio files
TEMP_DIR = "temp_audio_files"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'version': '1.0'
    })

@app.route('/download', methods=['POST'])
def download_subtitles():
    global SELECTED_MODEL
    try:
        video_url = request.form['url']
        model_name = request.form.get('model', SELECTED_MODEL)
        # Flag to indicate if we should force transcription even if subtitles are found
        force_transcribe = request.form.get('force_transcribe', 'false').lower() == 'true'
        language = request.form.get('language', 'auto')
        SELECTED_MODEL = model_name
        
        result = {'status': 'success', 'files': [], 'message': '', 'transcript': '', 'source': ''}

        # Validate URL format
        if not is_valid_video_url(video_url):
            return jsonify({'status': 'error', 'message': 'Invalid YouTube URL format'})

        # If force_transcribe is True, skip the subtitle fetching steps and go straight to generating
        if force_transcribe:
            logging.info(f"Force transcription requested for {video_url}, skipping subtitle checks")
            transcript = generate_subtitles_from_audio(video_url, model_name, language)
            if transcript:
                result['transcript'] = transcript
                result['message'] = f'Subtitles generated from audio successfully using Whisper ({model_name})!'
                result['source'] = 'generated'
                logging.info("Subtitles generated successfully")
                clean_temp_files()
                return jsonify(result)
            else:
                result['status'] = 'error'
                result['message'] = 'Failed to generate subtitles from audio. Please try a different model or check the video.'
                logging.error("Failed to generate subtitles")
                clean_temp_files()
                return jsonify(result)

        # Step 1: Try fetching from YouTube's "Show transcript"
        logging.info(f"Checking 'Show transcript' for {video_url}")
        transcript = fetch_youtube_transcript(video_url)
        if transcript:
            result['transcript'] = transcript
            result['message'] = 'Subtitles successfully fetched from YouTube "Show transcript"!'
            result['source'] = 'youtube'
            logging.info("Found 'Show transcript' subtitles")
            return jsonify(result)
 
        # Step 2: Try downloading subtitle files with yt-dlp
        logging.info("Attempting to download subtitle files with yt-dlp")
        output_file = os.path.join(app.static_folder, "subtitles")
        command = ["yt-dlp", "--write-sub", "--skip-download", "-o", output_file, video_url]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        # Check for subtitle files in the static directory
        for file in os.listdir(app.static_folder):
            if file.startswith("subtitles") and (file.endswith(".srt") or file.endswith(".vtt")):
                result['files'].append(file)

        if result['files']:
            result['message'] = 'Subtitle files downloaded successfully!'
            result['source'] = 'subtitles'
            logging.info("Subtitle files downloaded successfully")
            return jsonify(result)

        # If we reach here, no subtitles were found
        if not force_transcribe:
            # Return a message indicating no subtitles were found instead of transcribing
            result['status'] = 'not_found'
            result['message'] = 'No existing subtitles found. Use "Generate New Transcript" to create subtitles.'
            logging.info("No subtitles found and force_transcribe is False")
            return jsonify(result)
        
        # Step 3: Only generate subtitles if force_transcribe is True
        logging.info("Forced transcription requested, generating from audio")
        transcript = generate_subtitles_from_audio(video_url, model_name, language)
        if transcript:
            result['transcript'] = transcript
            result['message'] = f'Subtitles generated from audio successfully using Whisper ({model_name})!'
            result['source'] = 'generated'
            logging.info("Subtitles generated successfully")
        else:
            result['status'] = 'error'
            result['message'] = 'Failed to generate subtitles from audio. Please try a different model or check the video.'
            logging.error("Failed to generate subtitles")

        # Clean up temporary files
        clean_temp_files()

        return jsonify(result)

    except Exception as e:
        logging.error(f"Download failed: {str(e)}")
        # Clean up temporary files even on error
        clean_temp_files()
        return jsonify({'status': 'error', 'message': f'Download failed: {str(e)}'})
    
def is_valid_video_url(url):
    """Check if the URL is a valid YouTube URL"""
    return url.startswith(("https://www.youtube.com/watch", 
                          "https://youtube.com/watch",
                          "https://youtu.be/",
                          "https://www.youtube.com/shorts/",
                          "https://youtube.com/shorts/"))

def fetch_youtube_transcript(video_url):
    """Fetch subtitles from YouTube's 'Show transcript' and add timestamps"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(video_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script')
        for script in scripts:
            if 'captionTracks' in str(script):
                script_content = str(script)
                start = script_content.find('ytInitialPlayerResponse = ') + 25
                end = script_content.rfind('};') + 1
                json_data = script_content[start:end]
                try:
                    data = json.loads(json_data)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON parsing error: {str(e)}")
                    continue

                caption_tracks = data.get('captions', {}).get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
                if not caption_tracks:
                    return None

                transcript_url = caption_tracks[0]['baseUrl']
                transcript_response = requests.get(transcript_url)
                transcript_response.raise_for_status()

                transcript_soup = BeautifulSoup(transcript_response.text, 'lxml-xml')
                texts = transcript_soup.find_all('text')
                
                # Format transcript with timestamps
                transcript_with_timestamps = []
                for text in texts:
                    # Get the start time in seconds and convert to HH:MM:SS format
                    start_seconds = float(text.get('start', 0))
                    duration = float(text.get('dur', 0))
                    end_seconds = start_seconds + duration
                    
                    # Format timestamps
                    start_timestamp = format_timestamp(start_seconds)
                    end_timestamp = format_timestamp(end_seconds)
                    
                    # Add formatted text with timestamp
                    timestamp = f"[{start_timestamp} --> {end_timestamp}]"
                    transcript_with_timestamps.append(f"{timestamp} {text.get_text().strip()}")
                
                return "\n".join(transcript_with_timestamps)

        return None

    except Exception as e:
        logging.error(f"Failed to fetch subtitles: {str(e)}")
        return None


def generate_subtitles_from_audio(video_url, model_name="small", language="auto"):
    """Generate subtitles from video audio using Whisper, handling long audio"""
    try:
        # 加载模型 (保持不变)
        if model_name not in WHISPER_MODELS:
            logging.info(f"Loading Whisper model: {model_name}")
            WHISPER_MODELS[model_name] = whisper.load_model(model_name)
        
        # 下载音频 (保持不变)
        audio_file = os.path.join(TEMP_DIR, "temp_audio.mp3")
        command = [
            "yt-dlp",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", audio_file,
            video_url
        ]
        logging.info(f"Downloading audio from {video_url}")
        process = subprocess.run(command, capture_output=True, text=True)
        
        # 错误检查 (保持不变)
        if process.returncode != 0:
            logging.error(f"Failed to download audio: {process.stderr}")
            return None
        
        if not os.path.exists(audio_file):
            logging.error(f"Audio file not found after download: {audio_file}")
            return None
            
        file_size = os.path.getsize(audio_file)
        logging.info(f"Downloaded audio file size: {file_size} bytes")
        
        if file_size == 0:
            logging.error("Downloaded audio file is empty")
            return None

        # 加载音频 (保持不变)
        try:
            audio = AudioSegment.from_mp3(audio_file)
            audio_length_ms = len(audio)
            logging.info(f"Audio loaded successfully. Length: {audio_length_ms / 1000 / 60:.2f} minutes")
            
            if audio_length_ms == 0:
                logging.error("Audio has zero length")
                return None
        except Exception as e:
            logging.error(f"Failed to load audio file: {str(e)}")
            return None

        transcript_segments = []
        model = WHISPER_MODELS[model_name]
        
        segment_start_time = 0  # 跟踪每个片段的开始时间（秒）

        # 分段处理
        for start_ms in range(0, audio_length_ms, SEGMENT_LENGTH_MS):
            end_ms = min(start_ms + SEGMENT_LENGTH_MS, audio_length_ms)
            segment = audio[start_ms:end_ms]
            segment_file = os.path.join(TEMP_DIR, f"temp_audio_segment_{start_ms // 1000}.wav")
            
            # 导出段落
            segment.export(
                segment_file, 
                format="wav", 
                parameters=["-ar", "16000", "-ac", "1"]
            )
            
            segment_size = os.path.getsize(segment_file)
            logging.info(f"Processing segment {start_ms // 1000}s to {end_ms // 1000}s (Size: {segment_size} bytes)")
            
            if segment_size == 0:
                logging.error(f"Segment file is empty: {segment_file}")
                transcript_segments.append(f"[{format_timestamp(start_ms // 1000)}] [Empty audio segment]")
                continue

            # 转录段落
            try:
                # 使用指定语言或自动检测
                transcribe_options = {
                    "task": "transcribe", 
                    "verbose": True,  # 启用详细输出以获取段落级别的信息
                }
                if language != "auto":
                    transcribe_options["language"] = language
                
                # 使用 Whisper 的详细输出获取时间戳和段落
                result = model.transcribe(segment_file, **transcribe_options)
                
                if result and "segments" in result and result["segments"]:
                    # 处理每个带时间戳的段落
                    for segment_result in result["segments"]:
                        # 计算绝对时间戳（考虑当前片段在整个音频中的位置）
                        abs_start = segment_start_time + segment_result["start"]
                        abs_end = segment_start_time + segment_result["end"]
                        
                        # 格式化带时间戳的文本行
                        timestamp = f"[{format_timestamp(abs_start)} --> {format_timestamp(abs_end)}]"
                        text = segment_result["text"].strip()
                        
                        if text:  # 如果文本不为空
                            transcript_segments.append(f"{timestamp} {text}")
                            
                    logging.info(f"Successfully transcribed segment with {len(result['segments'])} segments")
                else:
                    logging.warning(f"Empty transcription for segment {start_ms // 1000}s to {end_ms // 1000}s")
                    transcript_segments.append(f"[{format_timestamp(start_ms // 1000)}] [Inaudible segment]")
            except Exception as e:
                logging.error(f"Error transcribing segment {start_ms // 1000}s to {end_ms // 1000}s: {str(e)}")
                transcript_segments.append(f"[{format_timestamp(start_ms // 1000)}] [Transcription error]")
            
            # 更新下一个片段的开始时间（秒）
            segment_start_time += (end_ms - start_ms) / 1000

        # 合并所有段落，每个段落一行
        full_transcript = "\n".join(transcript_segments)
        if not full_transcript.strip():
            logging.error("Generated transcript is empty")
            return None
            
        return full_transcript

    except Exception as e:
        logging.error(f"Failed to generate subtitles: {str(e)}")
        return None

def format_timestamp(seconds):
    """将秒数格式化为 HH:MM:SS 格式的时间戳"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"



def clean_temp_files():
    """Clean up all temporary files"""
    try:
        for file in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logging.info(f"Cleaned up {file_path}")
            except Exception as e:
                logging.error(f"Error cleaning up {file_path}: {str(e)}")
                
        # Also check the current directory for any temp files
        for file in os.listdir("."):
            if file.startswith("temp_audio") and (file.endswith(".mp3") or file.endswith(".wav")):
                os.remove(file)
                logging.info(f"Cleaned up {file}")
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

@app.route('/models', methods=['GET'])
def get_models():
    """Return available Whisper models"""
    models = ["tiny", "base", "small", "medium", "large"]
    return jsonify({
        'models': models,
        'selected': SELECTED_MODEL
    })

@app.route('/select_model', methods=['POST'])
def select_model():
    """Change the selected Whisper model"""
    global SELECTED_MODEL
    try:
        model_name = request.form['model']
        if model_name in ["tiny", "base", "small", "medium", "large"]:
            SELECTED_MODEL = model_name
            return jsonify({'status': 'success', 'message': f'Model changed to {model_name}'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid model name'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/languages', methods=['GET'])
def get_languages():
    """Return supported languages list"""
    # Main languages supported by Whisper
    languages = {
        "auto": "Auto detect",
        "zh": "Chinese",
        "en": "English",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
        "es": "Spanish",
        "ru": "Russian"
    }
    return jsonify(languages)

@app.route('/clear_temp', methods=['POST'])
def clear_temp():
    """Manually clear temporary files"""
    try:
        clean_temp_files()
        return jsonify({'status': 'success', 'message': 'Temporary files cleared'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Shutdown hook to clean temporary directory when app is stopped
def cleanup_on_shutdown():
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logging.info(f"Removed temporary directory: {TEMP_DIR}")
    except Exception as e:
        logging.error(f"Error removing temporary directory: {str(e)}")

# Register cleanup function
import atexit
atexit.register(cleanup_on_shutdown)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)