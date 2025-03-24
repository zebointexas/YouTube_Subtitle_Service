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
import whisper  # 导入whisper库进行本地处理

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Temporary directory for audio files
TEMP_DIR = "temp_audio_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Gemini API configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyBgLyImS_akhGoVaGYxioyv4DtydApFKnk")
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini model selection
GEMINI_MODEL = "gemini-1.5-flash"

# Audio segment length (in milliseconds, 10 minutes per segment)
SEGMENT_LENGTH_MS = 10 * 60 * 1000  # 10 minutes

# 本地Whisper模型配置
WHISPER_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large"
}

# 已加载的模型缓存
loaded_whisper_models = {}

SELECTED_MODEL = "small"  # 默认使用small模型

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
    """Generate an intelligent and flexible summary based on input content characteristics"""
    try:
        # Clean and prepare the input text
        lines = set()
        for line in transcript.split('\n'):
            # Handle both timestamp formats: [00:00:00] and [00:00:00 --> 00:00:00]
            if '] ' in line:
                text = line.split('] ', 1)[1].strip()
                if text:
                    lines.add(text)
        
        clean_text = " ".join(lines)
        
        if not clean_text.strip():
            return "No content to summarize."

        # Analyze input text characteristics
        input_length = len(clean_text)
        word_count = len(clean_text.split())
        avg_word_length = input_length / max(1, word_count)
        
        logging.info(f"Input analysis: {input_length} chars, {word_count} words, {avg_word_length:.2f} avg word length")

        # Truncate if exceeding Gemini context limits
        if input_length > 1500000:
            logging.warning("Input exceeds recommended length, truncating to 1.5M characters")
            clean_text = clean_text[:1500000]
            word_count = len(clean_text.split())

        # Dynamic summary sizing parameters based on content length and complexity
        # Shorter content gets proportionally longer summaries (higher %)
        if word_count < 1000:  # Very short content (~5 mins of speech)
            summary_ratio = 0.25  # 25% of original
            min_chars = 300
            max_chars = 1000
        elif word_count < 3000:  # Medium content (~15 mins of speech)
            summary_ratio = 0.18  # 18% of original
            min_chars = 500
            max_chars = 2000
        elif word_count < 10000:  # Longer content (~50 mins speech)
            summary_ratio = 0.12  # 12% of original
            min_chars = 1000
            max_chars = 3500
        else:  # Very long content (1+ hour speech)
            summary_ratio = 0.08  # 8% of original
            min_chars = 1500
            max_chars = 5000
        
        # Calculate target length within constraints
        target_length = max(min_chars, min(max_chars, int(input_length * summary_ratio)))
        target_words = int(target_length / avg_word_length)
        
        logging.info(f"Target summary: ~{target_length} chars, ~{target_words} words ({summary_ratio*100:.1f}% of original)")

        # Create Gemini model instance
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Analyze content type to determine summarization approach
        is_technical = check_if_technical(clean_text)
        is_narrative = check_if_narrative(clean_text)
        content_language = detect_language(clean_text)
        
        # Build dynamic prompt based on content analysis
        system_instruction = (
            "You are an expert summarizer with the ability to adapt to different content types. "
            "Create a concise yet comprehensive summary in Chinese language that captures the essential information, "
            "regardless of the source language."
            "Both titles and content should be in pure Chinese characters only, without adding pinyin annotations."
        )
        
        format_instruction = "Organize the summary with clear structure using appropriate headings and paragraphs in Chinese. "
        
        if is_technical:
            content_guidance = (
                "This appears to be technical content. Preserve key technical details, definitions, and processes. "
                "Maintain technical accuracy while making complex concepts accessible. "
                "Use bullet points for steps or specifications if appropriate. "
                "The summary should be in Chinese regardless of the source language."
            )
        elif is_narrative:
            content_guidance = (
                "This appears to be narrative content. Capture the main storyline, key events, and important themes. "
                "Preserve the narrative flow while condensing repetitive elements. "
                "Maintain the emotional tone of the original. "
                "The summary should be in Chinese regardless of the source language."
            )
        else:
            content_guidance = (
                "Focus on extracting the main points, key supporting details, and overall message. "
                "Identify and include important examples, statistics or quotes that substantiate main points. "
                "The summary should be in Chinese regardless of the source language."
            )
        
        length_instruction = (
            f"Aim for approximately {target_length} characters (about {target_words} words), "
            f"which is approximately {summary_ratio*100:.1f}% of the original text length. "
            "Adjust the level of detail appropriately to meet this target while preserving key information. "
        )
        
        # Combine instructions into full prompt
        prompt = (
            f"{system_instruction}\n\n"
            f"{content_guidance}\n"
            f"{format_instruction}\n"
            f"{length_instruction}\n\n"
            "TEXT TO SUMMARIZE:\n\n"
            f"{clean_text}"
        )

        # Generate summary
        logging.info(f"Requesting Chinese summary from Gemini ({GEMINI_MODEL})")
        response = model.generate_content(prompt)
        
        if response and response.text:
            summary = response.text.strip()
            summary_length = len(summary)
            summary_words = len(summary.split())
            
            logging.info(f"Generated Chinese summary: {summary_length} chars, {summary_words} words")

            # Check if summary is significantly off-target in length (±40%)
            if not (target_length * 0.6 <= summary_length <= target_length * 1.4):
                logging.warning(f"Summary length {summary_length} deviates from target {target_length}, adjusting...")
                
                # Adjust strategy based on whether summary is too long or too short
                if summary_length > target_length * 1.4:
                    adjustment_prompt = (
                        "The Chinese summary you provided is too long. Please condense it further while preserving all key points. "
                        f"The target length is approximately {target_length} characters. "
                        "Focus on the most important information and remove unnecessary details. "
                        "Keep the summary in Chinese."
                    )
                else:
                    adjustment_prompt = (
                        "The Chinese summary you provided is too brief. Please expand it with more details while maintaining conciseness. "
                        f"The target length is approximately {target_length} characters. "
                        "Include more context and supporting details for important points. "
                        "Keep the summary in Chinese."
                    )
                
                # Send adjustment request with original summary
                adjustment_query = f"{adjustment_prompt}\n\nCURRENT SUMMARY:\n{summary}"
                retry_response = model.generate_content(adjustment_query)
                
                if retry_response and retry_response.text:
                    adjusted_summary = retry_response.text.strip()
                    adjusted_length = len(adjusted_summary)
                    adjusted_words = len(adjusted_summary.split())
                    
                    logging.info(f"Adjusted Chinese summary: {adjusted_length} chars, {adjusted_words} words")
                    
                    # Use adjusted summary if it's closer to target
                    if abs(adjusted_length - target_length) < abs(summary_length - target_length):
                        summary = adjusted_summary

            return summary
        else:
            logging.error("No summary returned from Gemini API")
            return "Summary generation failed."

    except Exception as e:
        logging.error(f"Summary generation failed with Gemini API: {str(e)}")
        return f"Error generating summary: {str(e)}"

# Helper functions for content analysis

def check_if_technical(text):
    """Determine if content appears to be technical in nature"""
    # Technical indicators: high frequency of specialized terms, numbers, measurements
    technical_indicators = [
        'algorithm', 'technical', 'function', 'data', 'system', 'method',
        'analysis', 'equation', 'process', 'procedure', 'implementation',
        'configuration', 'parameters', 'specification', 'architecture',
        'component', 'device', 'module', 'interface', 'protocol', 'framework'
    ]
    
    # Count technical terms
    word_count = len(text.split())
    technical_count = sum(1 for indicator in technical_indicators if indicator.lower() in text.lower())
    
    # Check for numerical density (percentage of words that contain numbers)
    words = text.split()
    numerical_count = sum(1 for word in words if any(char.isdigit() for char in word))
    numerical_density = numerical_count / max(1, word_count)
    
    # Return true if either technical term density is high or numerical density is high
    return (technical_count / max(1, word_count) > 0.01) or (numerical_density > 0.05)

def check_if_narrative(text):
    """Determine if content appears to be narrative/storytelling in nature"""
    # Narrative indicators: personal pronouns, past tense verbs, time references
    narrative_indicators = [
        ' I ', ' me ', ' my ', ' we ', ' our ', ' us ',
        'story', 'experience', 'journey', 'remember', 'recall',
        'happened', 'occurred', 'felt', 'thought', 'believed',
        'said', 'told', 'asked', 'replied', 'answered',
        'yesterday', 'last week', 'years ago', 'once', 'when'
    ]
    
    # Count narrative markers
    narrative_count = sum(1 for indicator in narrative_indicators if f' {indicator.lower()} ' in f' {text.lower()} ')
    word_count = len(text.split())
    
    # Return true if narrative markers density is moderately high
    return narrative_count / max(1, word_count) > 0.015

def detect_language(text):
    """Simple language detection"""
    # This is a simplified detection - in a full implementation, you might use a language detection library
    # We're checking for common characters in different scripts as a basic proxy
    
    # Sample first 1000 characters for efficiency
    sample = text[:1000].lower()
    
    # Check for Chinese characters
    if any('\u4e00' <= char <= '\u9fff' for char in sample):
        return 'zh'
    
    # Check for Japanese-specific characters (hiragana, katakana)
    if any('\u3040' <= char <= '\u30ff' for char in sample):
        return 'ja'
    
    # Check for Korean hangul
    if any('\uac00' <= char <= '\ud7a3' for char in sample):
        return 'ko'
    
    # Check for Cyrillic (Russian and related)
    if any('\u0400' <= char <= '\u04ff' for char in sample):
        return 'ru'
    
    # Default to 'en' (English) or other Latin-script languages
    return 'en'

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

def get_whisper_model(model_name="small"):
    """获取或加载Whisper模型"""
    global loaded_whisper_models
    
    if model_name not in loaded_whisper_models:
        logging.info(f"Loading Whisper model: {model_name}")
        try:
            # 加载指定的模型
            loaded_whisper_models[model_name] = whisper.load_model(model_name)
            logging.info(f"Successfully loaded Whisper model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load Whisper model {model_name}: {str(e)}")
            # 如果加载失败，尝试加载基础模型
            if model_name != "base":
                logging.info("Trying to load base model instead")
                try:
                    loaded_whisper_models[model_name] = whisper.load_model("base")
                    return loaded_whisper_models[model_name]
                except:
                    logging.error("Could not load any Whisper model")
                    return None
            return None
    
    return loaded_whisper_models[model_name]

def generate_subtitles_from_audio(video_url, model_name="small", language="auto"):
    """Generate subtitles from video audio using local Whisper model"""
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
            
        # 获取Whisper模型
        model = get_whisper_model(model_name)
        if model is None:
            logging.error(f"Failed to get Whisper model: {model_name}")
            return None
            
        # 设置转写参数
        whisper_options = {
            "verbose": False,
            "task": "transcribe"
        }
        
        # 如果指定了语言，则添加到选项中
        if language != "auto":
            whisper_options["language"] = language
        
        # 转写音频
        logging.info(f"Transcribing audio with Whisper model: {model_name}")
        result = model.transcribe(audio_file, **whisper_options)
        
        if result and "segments" in result:
            # 格式化转写结果，添加时间戳
            transcript_segments = []
            for segment in result["segments"]:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()
                
                if text:
                    # 格式化时间戳
                    start_timestamp = format_timestamp(start_time)
                    end_timestamp = format_timestamp(end_time)
                    timestamp = f"[{start_timestamp} --> {end_timestamp}]"
                    transcript_segments.append(f"{timestamp} {text}")
            
            full_transcript = "\n".join(transcript_segments)
            if not full_transcript.strip():
                logging.error("Generated transcript is empty")
                return None
                
            return full_transcript
        else:
            logging.error("No segments found in transcription result")
            return None

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