from flask import Flask, request, jsonify, render_template
import subprocess
import os
import requests
from bs4 import BeautifulSoup
import whisper
from pydub import AudioSegment
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Available Whisper models
WHISPER_MODELS = {}
SELECTED_MODEL = "small"  # Default to small, change to "medium" for better accuracy

# Audio segment length (in milliseconds, 30 minutes per segment)
SEGMENT_LENGTH_MS = 30 * 60 * 1000  # 30 minutes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download', methods=['POST'])
def download_subtitles():
    global SELECTED_MODEL
    try:
        video_url = request.form['url']
        model_name = request.form.get('model', SELECTED_MODEL)
        SELECTED_MODEL = model_name
        
        result = {'status': 'success', 'files': [], 'message': '', 'transcript': ''}

        # Step 1: Try fetching from YouTube's "Show transcript"
        logging.info(f"Checking 'Show transcript' for {video_url}")
        transcript = fetch_youtube_transcript(video_url)
        if transcript:
            result['transcript'] = transcript
            result['message'] = 'Subtitles successfully fetched from YouTube "Show transcript"!'
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
            logging.info("Subtitle files downloaded successfully")
        else:
            # Step 3: Generate subtitles from audio using Whisper
            logging.info("No subtitles found, generating from audio")
            transcript = generate_subtitles_from_audio(video_url, model_name)
            if transcript:
                result['transcript'] = transcript
                result['message'] = f'Subtitles generated from audio successfully using Whisper ({model_name})!'
                logging.info("Subtitles generated successfully")
            else:
                result['status'] = 'error'
                result['message'] = 'No subtitles available and failed to generate from audio.'
                logging.error("Failed to generate subtitles")

        if stderr:
            result['message'] += f"\nError message: {stderr}"
            logging.warning(f"yt-dlp stderr: {stderr}")

        # Clean up temporary files
        for file in os.listdir("."):
            if file.startswith("temp_audio") and (file.endswith(".mp3") or file.endswith(".wav")):
                os.remove(file)
                logging.info(f"Cleaned up {file}")

        return jsonify(result)

    except Exception as e:
        logging.error(f"Download failed: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Download failed: {str(e)}'})

def fetch_youtube_transcript(video_url):
    """Fetch subtitles from YouTube's 'Show transcript'"""
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
                import json
                data = json.loads(json_data)

                caption_tracks = data.get('captions', {}).get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
                if not caption_tracks:
                    return None

                transcript_url = caption_tracks[0]['baseUrl']
                transcript_response = requests.get(transcript_url)
                transcript_response.raise_for_status()

                transcript_soup = BeautifulSoup(transcript_response.text, 'lxml-xml')
                texts = transcript_soup.find_all('text')
                transcript = '\n'.join([text.get_text() for text in texts])
                return transcript

        return None

    except Exception as e:
        logging.error(f"Failed to fetch subtitles: {str(e)}")
        return None

def generate_subtitles_from_audio(video_url, model_name="small"):
    """Generate subtitles from video audio using Whisper, handling long audio"""
    try:
        # Load model on demand
        if model_name not in WHISPER_MODELS:
            logging.info(f"Loading Whisper model: {model_name}")
            WHISPER_MODELS[model_name] = whisper.load_model(model_name)
        
        # Download audio using yt-dlp
        audio_file = "temp_audio.mp3"
        command = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "-o", audio_file,
            video_url
        ]
        logging.info(f"Downloading audio from {video_url}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if not os.path.exists(audio_file):
            logging.error(f"Failed to download audio: {stderr}")
            return None

        # Load audio and split into segments
        audio = AudioSegment.from_mp3(audio_file)
        audio_length_ms = len(audio)
        logging.info(f"Audio length: {audio_length_ms / 1000 / 60:.2f} minutes")

        transcript = []
        model = WHISPER_MODELS[model_name]

        # Process in 30-minute segments to avoid memory issues
        for start_ms in range(0, audio_length_ms, SEGMENT_LENGTH_MS):
            end_ms = min(start_ms + SEGMENT_LENGTH_MS, audio_length_ms)
            segment = audio[start_ms:end_ms]
            segment_file = f"temp_audio_segment_{start_ms // 1000}.wav"
            segment.export(segment_file, format="wav")
            logging.info(f"Processing segment {start_ms // 1000}s to {end_ms // 1000}s")

            # Transcribe segment
            result = model.transcribe(segment_file)
            transcript.append(result["text"])

            # Clean up segment file
            os.remove(segment_file)

        # Combine all segments
        full_transcript = "\n".join(transcript)
        return full_transcript if full_transcript else None

    except Exception as e:
        logging.error(f"Failed to generate subtitles: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True)