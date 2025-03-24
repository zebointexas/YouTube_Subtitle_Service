from flask import Flask, request, jsonify, render_template
import subprocess
import os
import requests
from bs4 import BeautifulSoup
import logging
import json
import shutil
import time
import base64
from pydub import AudioSegment
import google.generativeai as genai

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Temporary directory for audio files
TEMP_DIR = "temp_audio_files"
os.makedirs(TEMP_DIR, exist_ok=True)


# Gemini API configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyBgLyImS_akhGoVaGYxioyv4DtydApFKnk")  # 请替换为你的 Gemini API 密钥
genai.configure(api_key=GOOGLE_API_KEY)

GEMINI_MODEL = "gemini-1.5-flash"  # 或 "gemini-1.5-pro" 如果可用

# Gemini model selection (免费层级可用 "gemini-1.5-flash" 或 "gemini-1.5-pro"，视可用性而定)
GEMINI_MODEL = "gemini-1.5-flash"  # 免费且支持大上下文，建议使用

# Audio segment length (in milliseconds, 10 minutes per segment)
SEGMENT_LENGTH_MS = 10 * 60 * 1000  # 10 minutes

# [其他路由保持不变，如 /, /api/health, /download, /models 等]
 
SELECTED_MODEL = "small"  # Default to small

 




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
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
        force_transcribe = request.form.get('force_transcribe', 'false').lower() == 'true'
        language = request.form.get('language', 'auto')
        SELECTED_MODEL = model_name
        
        result = {'status': 'success', 'files': [], 'message': '', 'transcript': '', 'source': '', 'summary': ''}

        if not is_valid_video_url(video_url):
            return jsonify({'status': 'error', 'message': 'Invalid YouTube URL format'})

        if force_transcribe:
            logging.info(f"Force transcription requested for {video_url}, skipping subtitle checks")
            transcript = generate_subtitles_from_audio(video_url, model_name, language)
            if transcript:
                result['transcript'] = transcript
                result['message'] = f'Subtitles generated from audio successfully using Whisper ({model_name})!'
                result['source'] = 'generated'
                result['summary'] = generate_summary(transcript)  # Generate summary
                logging.info("Subtitles and summary generated successfully")
                clean_temp_files()
                return jsonify(result)
            else:
                result['status'] = 'error'
                result['message'] = 'Failed to generate subtitles from audio. Please try a different model or check the video.'
                logging.error("Failed to generate subtitles")
                clean_temp_files()
                return jsonify(result)

        # Try fetching from YouTube's "Show transcript"
        logging.info(f"Checking 'Show transcript' for {video_url}")
        transcript = fetch_youtube_transcript(video_url)
        if transcript:
            result['transcript'] = transcript
            result['message'] = 'Subtitles successfully fetched from YouTube "Show transcript"!'
            result['source'] = 'youtube'
            result['summary'] = generate_summary(transcript)  # Generate summary
            logging.info("Found 'Show transcript' subtitles and generated summary")
            return jsonify(result)
 
        # Try downloading subtitle files with yt-dlp
        logging.info("Attempting to download subtitle files with yt-dlp")
        output_file = os.path.join(app.static_folder, "subtitles")
        command = ["yt-dlp", "--write-sub", "--skip-download", "-o", output_file, video_url]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        for file in os.listdir(app.static_folder):
            if file.startswith("subtitles") and (file.endswith(".srt") or file.endswith(".vtt")):
                result['files'].append(file)

        if result['files']:
            result['message'] = 'Subtitle files downloaded successfully!'
            result['source'] = 'subtitles'
            # For simplicity, we'll need to read the subtitle file to generate a summary
            with open(os.path.join(app.static_folder, result['files'][0]), 'r', encoding='utf-8') as f:
                transcript = f.read()
                result['transcript'] = transcript
                result['summary'] = generate_summary(transcript)
            logging.info("Subtitle files downloaded and summary generated successfully")
            return jsonify(result)

        if not force_transcribe:
            result['status'] = 'not_found'
            result['message'] = 'No existing subtitles found. Use "Generate New Transcript" to create subtitles.'
            logging.info("No subtitles found and force_transcribe is False")
            return jsonify(result)
        
        logging.info("Forced transcription requested, generating from audio")
        transcript = generate_subtitles_from_audio(video_url, model_name, language)
        if transcript:
            result['transcript'] = transcript
            result['message'] = f'Subtitles generated from audio successfully using Whisper ({model_name})!'
            result['source'] = 'generated'
            result['summary'] = generate_summary(transcript)  # Generate summary
            logging.info("Subtitles and summary generated successfully")
        else:
            result['status'] = 'error'
            result['message'] = 'Failed to generate subtitles from audio. Please try a different model or check the video.'
            logging.error("Failed to generate subtitles")

        clean_temp_files()
        return jsonify(result)

    except Exception as e:
        logging.error(f"Download failed: {str(e)}")
        clean_temp_files()
        return jsonify({'status': 'error', 'message': f'Download failed: {str(e)}'})


def generate_summary(transcript):
    """使用 Gemini API 生成动态长度的详细文本总结"""
    try:
        # 清理输入，保留所有文本
        lines = set()
        for line in transcript.split('\n'):
            if '] ' in line:
                text = line.split('] ', 1)[1].strip()
                if text:
                    lines.add(text)
        clean_text = " ".join(lines)
        
        if not clean_text.strip():
            return "No content to summarize."

        input_length = len(clean_text)
        logging.info(f"Input length: {input_length} characters")

        # 如果文本超长，截断到 Gemini 支持的最大上下文（保守估计 1.5M 字符）
        if input_length > 1500000:
            logging.warning("Input exceeds recommended length, truncating to 1.5M characters")
            clean_text = clean_text[:1500000]

        # 根据输入长度动态设定目标总结长度（字符数）
        # 目标长度为输入的 10%-20%，但设置最小 500 字符（约 100 字），最大 5000 字符（约 1000 字）
        target_length = max(500, min(5000, int(input_length * 0.15)))  # 15% 为基准
        logging.info(f"Target summary length: {target_length} characters")

        # 创建 Gemini 模型实例
        model = genai.GenerativeModel(GEMINI_MODEL)

        # 设计动态提示，要求根据内容调整长度并保持结构化
        prompt = (
            "You are an expert summarizer. Summarize the following text in a detailed and structured way. "
            f"Adjust the summary length to approximately {target_length} characters (about {target_length // 5} words), "
            "ensuring it captures the main points, key details, and overall narrative proportionally to the input length. "
            "Organize the summary into clear sections (e.g., introduction, main content, conclusion) and reflect the tone and intent of the original text. "
            "Do not omit critical information and provide an accurate, engaging overview:\n\n"
            f"{clean_text}"
        )

        # 调用 Gemini API
        logging.info("Sending request to Gemini API for dynamic summarization")
        response = model.generate_content(prompt)
        
        if response and response.text:
            summary = response.text.strip()
            summary_length = len(summary)
            logging.info(f"Generated summary length: {summary_length} characters")

            # 如果总结长度偏差过大（±50%），尝试重新生成
            if not (target_length * 0.5 <= summary_length <= target_length * 1.5):
                logging.warning(f"Summary length {summary_length} deviates from target {target_length}, retrying")
                retry_prompt = (
                    "The previous summary length was not suitable. "
                    f"Please summarize the following text with a target length of approximately {target_length} characters, "
                    "ensuring all key points are covered with appropriate depth and structure:\n\n"
                    f"{clean_text}"
                )
                retry_response = model.generate_content(retry_prompt)
                if retry_response and retry_response.text:
                    summary = retry_response.text.strip()
                    summary_length = len(summary)
                    logging.info(f"Retry summary length: {summary_length} characters")

            return summary
        else:
            logging.error("No summary returned from Gemini API")
            return "Summary generation failed."

    except Exception as e:
        logging.error(f"Summary generation failed with Gemini API: {str(e)}")
        return f"Error generating summary: {str(e)}"
    

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
    """Generate subtitles from video audio using Hugging Face Whisper API"""
    try:
        # Download audio
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
        
        # Error checking
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

        # Load audio
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

        # Get Hugging Face model path
        hf_model_path = WHISPER_MODELS.get(model_name, WHISPER_MODELS["small"])
        api_url = f"{HF_API_URL}{hf_model_path}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        transcript_segments = []
        segment_start_time = 0  # Track start time of each segment (seconds)

        # Process in segments
        for start_ms in range(0, audio_length_ms, SEGMENT_LENGTH_MS):
            end_ms = min(start_ms + SEGMENT_LENGTH_MS, audio_length_ms)
            segment = audio[start_ms:end_ms]
            segment_file = os.path.join(TEMP_DIR, f"temp_audio_segment_{start_ms // 1000}.wav")
            
            # Export segment
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

            # Transcribe using Hugging Face API
            try:
                with open(segment_file, "rb") as f:
                    data = f.read()
                    
                # Convert audio to base64 for API
                audio_bytes = base64.b64encode(data).decode("utf-8")
                
                # Prepare API parameters
                payload = {
                    "inputs": {
                        "audio": audio_bytes
                    },
                    "parameters": {
                        "return_timestamps": True
                    }
                }
                
                # Set language if specified
                if language != "auto":
                    payload["parameters"]["language"] = language
                
                # Submit transcription request
                logging.info(f"Sending audio segment to Hugging Face API ({hf_model_path})")
                response = requests.post(api_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract segments with timestamps from response
                    if "chunks" in result:
                        # New API response format with chunks
                        for chunk in result["chunks"]:
                            if "timestamp" in chunk and chunk.get("text"):
                                # Extract timestamp ranges
                                ts = chunk["timestamp"]
                                if isinstance(ts, list) and len(ts) == 2:
                                    # Convert timestamps and adjust for segment position
                                    abs_start = segment_start_time + ts[0]
                                    abs_end = segment_start_time + ts[1]
                                    
                                    # Format timestamp
                                    timestamp = f"[{format_timestamp(abs_start)} --> {format_timestamp(abs_end)}]"
                                    transcript_segments.append(f"{timestamp} {chunk['text'].strip()}")
                    elif isinstance(result, dict) and "text" in result:
                        # Simple response without timestamps
                        timestamp = f"[{format_timestamp(segment_start_time)}]"
                        transcript_segments.append(f"{timestamp} {result['text'].strip()}")
                else:
                    logging.error(f"API error: {response.status_code}, {response.text}")
                    transcript_segments.append(f"[{format_timestamp(segment_start_time)}] [API error: {response.status_code}]")
                    
            except Exception as e:
                logging.error(f"Error transcribing segment {start_ms // 1000}s to {end_ms // 1000}s: {str(e)}")
                transcript_segments.append(f"[{format_timestamp(start_ms // 1000)}] [Transcription error]")
            
            # Update start time for next segment (seconds)
            segment_start_time += (end_ms - start_ms) / 1000

        # Combine all segments
        full_transcript = "\n".join(transcript_segments)
        if not full_transcript.strip():
            logging.error("Generated transcript is empty")
            return None
            
        return full_transcript

    except Exception as e:
        logging.error(f"Failed to generate subtitles: {str(e)}")
        return None

def format_timestamp(seconds):
    """Format seconds to HH:MM:SS timestamp"""
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
    models = list(WHISPER_MODELS.keys())
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
        if model_name in WHISPER_MODELS:
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