from flask import Flask, request, jsonify
import subprocess
import os
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/download', methods=['POST'])
def download_subtitles():
    try:
        video_url = request.form['url']

        # Try to download subtitle files using yt-dlp
        output_file = "subtitles"
        command = [
            "yt-dlp",
            "--write-sub",
            "--skip-download",
            "-o", output_file,
            video_url
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        # Check if subtitle files were generated
        result = {'status': 'success', 'files': [], 'message': '', 'transcript': ''}
        for file in os.listdir("."):
            if file.startswith(output_file) and (file.endswith(".srt") or file.endswith(".vtt")):
                result['files'].append(file)

        # If yt-dlp didn't find subtitles, try fetching from YouTube's "Show transcript"
        if not result['files']:
            transcript = fetch_youtube_transcript(video_url)
            if transcript:
                result['transcript'] = transcript
                result['message'] = 'Subtitles successfully fetched from YouTube "Show transcript"!'
            else:
                result['status'] = 'error'
                result['message'] = 'No subtitles found. The video may not have subtitles or is inaccessible.'
        else:
            result['message'] = 'Subtitle files downloaded successfully!'

        if stderr:
            result['message'] += f"\nError message: {stderr}"

        return jsonify(result)

    except Exception as e:
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

                # Parse XML using lxml
                transcript_soup = BeautifulSoup(transcript_response.text, 'lxml-xml')  # Changed to 'lxml-xml'
                texts = transcript_soup.find_all('text')
                transcript = '\n'.join([text.get_text() for text in texts])
                return transcript

        return None

    except Exception as e:
        print(f"Failed to fetch subtitles: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True)