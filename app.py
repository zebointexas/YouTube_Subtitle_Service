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

        # 尝试使用 yt-dlp 下载字幕文件
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

        # 检查是否有字幕文件生成
        result = {'status': 'success', 'files': [], 'message': '', 'transcript': ''}
        for file in os.listdir("."):
            if file.startswith(output_file) and (file.endswith(".srt") or file.endswith(".vtt")):
                result['files'].append(file)

        # 如果 yt-dlp 没找到字幕，尝试抓取 YouTube 的 "Show transcript"
        if not result['files']:
            transcript = fetch_youtube_transcript(video_url)
            if transcript:
                result['transcript'] = transcript
                result['message'] = '从 YouTube "Show transcript" 获取字幕成功！'
            else:
                result['status'] = 'error'
                result['message'] = '未找到字幕，视频可能没有字幕或无法访问。'
        else:
            result['message'] = '字幕文件下载成功！'

        if stderr:
            result['message'] += f"\n错误信息: {stderr}"

        return jsonify(result)

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'下载失败: {str(e)}'})

def fetch_youtube_transcript(video_url):
    """从 YouTube 页面抓取 'Show transcript' 的字幕"""
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

                # 使用 lxml 解析 XML
                transcript_soup = BeautifulSoup(transcript_response.text, 'lxml-xml')  # 修改为 'lxml-xml'
                texts = transcript_soup.find_all('text')
                transcript = '\n'.join([text.get_text() for text in texts])
                return transcript

        return None

    except Exception as e:
        print(f"抓取字幕失败: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True)