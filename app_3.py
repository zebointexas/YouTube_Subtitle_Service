from flask import Flask, request, render_template, jsonify, send_file
import yt_dlp
import os
from datetime import datetime
import uuid
import webvtt
import whisper
import logging

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建存储目录
VIDEO_DIR = "static/videos"
AUDIO_DIR = "static/audio"
SUBTITLE_DIR = "static/subtitles"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SUBTITLE_DIR, exist_ok=True)

def convert_vtt_to_txt(vtt_file_path, include_timestamps=False):
    """将 .vtt 字幕文件转换为 .txt 格式，可选择是否包含时间戳，并去重"""
    txt_file_path = vtt_file_path.replace('.vtt', '.txt')
    
    try:
        vtt = webvtt.read(vtt_file_path)
        deduplicated_captions = []
        last_text = None
        last_end = None
        
        def time_to_seconds(time_str):
            h, m, s = map(float, time_str.replace(',', '.').split(':'))
            return h * 3600 + m * 60 + s
        
        for caption in vtt:
            current_text = caption.text.strip()
            current_start = time_to_seconds(caption.start)
            current_end = time_to_seconds(caption.end)
            
            if last_text == current_text and last_end and abs(current_start - last_end) < 1.0:
                if include_timestamps:
                    deduplicated_captions[-1]['end'] = caption.end
                logger.info(f"合并重复字幕: {current_text}")
            else:
                caption_data = {'text': current_text}
                if include_timestamps:
                    caption_data['start'] = caption.start
                    caption_data['end'] = caption.end
                deduplicated_captions.append(caption_data)
            last_text = current_text
            last_end = current_end
        
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            for caption in deduplicated_captions:
                if include_timestamps:
                    start_time = caption['start'].replace('.', ',')
                    end_time = caption['end'].replace('.', ',')
                    txt_file.write(f"{start_time} --> {end_time}: {caption['text']}\n")
                else:
                    txt_file.write(f"{caption['text']}\n")
        
        logger.info(f"字幕转换成功: {txt_file_path}")
        return txt_file_path
    except Exception as e:
        logger.error(f"转换 VTT 到 TXT 时出错: {e}")
        return None

def download_audio(youtube_url, output_file):
    """下载 YouTube 视频的音频并确保返回有效文件"""
    ydl_opts = {
        'format': 'bestaudio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_file}.%(ext)s',
        'noplaylist': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"开始下载音频: {youtube_url}")
            ydl.download([youtube_url])
            possible_extensions = ['.mp3', '.webm', '.m4a']
            for ext in possible_extensions:
                audio_file = f"{output_file}{ext}"
                if os.path.exists(audio_file):
                    logger.info(f"音频下载成功: {audio_file}")
                    if ext != '.mp3':
                        mp3_file = f"{output_file}.mp3"
                        os.system(f"ffmpeg -i {audio_file} -codec:a mp3 {mp3_file}")
                        if os.path.exists(mp3_file):
                            os.remove(audio_file)
                            logger.info(f"转换为 MP3: {mp3_file}")
                            return mp3_file
                        else:
                            logger.error(f"MP3 转换失败: {mp3_file}")
                            return None
                    return audio_file
            logger.error(f"未找到任何音频文件: {output_file}")
            return None
    except Exception as e:
        logger.error(f"下载音频时出错: {e}")
        return None

def download_video(youtube_url, output_file):
    """下载 YouTube 视频"""
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': f'{output_file}.%(ext)s',
        'noplaylist': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"开始下载视频: {youtube_url}")
            ydl.download([youtube_url])
            video_file = f"{output_file}.mp4"
            if os.path.exists(video_file):
                logger.info(f"视频下载成功: {video_file}")
                return video_file
            logger.error(f"视频文件未生成: {video_file}")
            return None
    except Exception as e:
        logger.error(f"下载视频时出错: {e}")
        return None

def transcribe_audio_to_txt(audio_file, include_timestamps=False, language="zh", model_size="small"):
    """使用 Whisper 将音频转换为 TXT 格式，支持不同模型"""
    txt_file_path = os.path.join(SUBTITLE_DIR, os.path.basename(audio_file).replace('.mp3', '.txt'))
    
    try:
        if not os.path.exists(audio_file):
            logger.error(f"音频文件不存在: {audio_file}")
            return None
        
        logger.info(f"加载 Whisper 模型 ({model_size})，语言: {language}")
        model = whisper.load_model(model_size)
        logger.info(f"开始转录音频: {audio_file}")
        result = model.transcribe(audio_file, language=language, fp16=False)
        
        if not result["segments"]:
            logger.error("转录结果为空")
            return None
        
        deduplicated_captions = []
        last_text = None
        last_end = None
        
        for segment in result["segments"]:
            current_text = segment['text'].strip()
            current_start = segment['start']
            current_end = segment['end']
            
            if last_text == current_text and last_end and abs(current_start - last_end) < 1.0:
                if include_timestamps:
                    deduplicated_captions[-1]['end'] = current_end
                logger.info(f"合并重复转录: {current_text}")
            else:
                caption_data = {'text': current_text}
                if include_timestamps:
                    caption_data['start'] = current_start
                    caption_data['end'] = current_end
                deduplicated_captions.append(caption_data)
            last_text = current_text
            last_end = current_end
        
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            for caption in deduplicated_captions:
                if include_timestamps:
                    start_time = f"{int(caption['start'] // 3600):02d}:{int((caption['start'] % 3600) // 60):02d}:{int(caption['start'] % 60):02d},{int((caption['start'] % 1) * 1000):03d}"
                    end_time = f"{int(caption['end'] // 3600):02d}:{int((caption['end'] % 3600) // 60):02d}:{int(caption['end'] % 60):02d},{int((caption['end'] % 1) * 1000):03d}"
                    txt_file.write(f"{start_time} --> {end_time}: {caption['text']}\n")
                else:
                    txt_file.write(f"{caption['text']}\n")
        
        logger.info(f"转录完成，生成文件: {txt_file_path}")
        return txt_file_path
    except Exception as e:
        logger.error(f"转录音频到 TXT 时出错: {e}")
        return None

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/download_subtitle', methods=['POST'])
def download_subtitle():
    """API 端点：下载 YouTube 字幕或转录音频，同时下载视频"""
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    include_timestamps = data.get('include_timestamps', False)
    model_size = data.get('model_size', 'small')  # 默认使用 small 模型
    
    if not youtube_url:
        return jsonify({'status': 'error', 'message': '未提供 YouTube URL'}), 400
    
    # 验证 model_size 是否有效
    valid_models = ['base', 'small', 'medium']
    if model_size not in valid_models:
        return jsonify({'status': 'error', 'message': f"无效的模型大小: {model_size}，支持: {valid_models}"}), 400
    
    unique_id = str(uuid.uuid4())[:8]
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_prefix = f"{date_string}_{unique_id}_"
    
    try:
        with yt_dlp.YoutubeDL({'skip_download': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'video')
            subtitles = info.get('subtitles', {})
        
        # 同时下载视频
        video_file = download_video(youtube_url, os.path.join(VIDEO_DIR, f"{output_prefix}{video_title}"))
        video_info = None
        if video_file:
            video_info = {
                'filename': os.path.basename(video_file),
                'path': f'/static/videos/{os.path.basename(video_file)}'
            }
        else:
            logger.warning("视频下载失败，但继续处理字幕")

        subtitle_files = []
        # 只检查手动上传的字幕，不使用自动生成的字幕
        if subtitles:
            logger.info(f"找到 '{video_title}' 的手动字幕")
            
            ydl_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': False,  # 确保不下载自动字幕
                'subtitlesformat': 'vtt',
                'outtmpl': os.path.join(SUBTITLE_DIR, f'{output_prefix}%(title)s.%(ext)s'),
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([youtube_url])
            
            for lang, sub_info in subtitles.items():
                subtitle_file = f"{output_prefix}{video_title}.{lang}.vtt"
                full_path = os.path.join(SUBTITLE_DIR, subtitle_file)
                
                if os.path.exists(full_path):
                    txt_file = convert_vtt_to_txt(full_path, include_timestamps=include_timestamps)
                    if txt_file:
                        subtitle_files.append({
                            'language': lang,
                            'filename': os.path.basename(txt_file),
                            'path': f'/static/subtitles/{os.path.basename(txt_file)}',
                            'type': 'manual'
                        })
        # 如果没有手动字幕，直接进行音频转录
        else:
            logger.info(f"'{video_title}' 无手动字幕，开始提取音频并转录为中文 (模型: {model_size})")
            audio_file = download_audio(youtube_url, os.path.join(AUDIO_DIR, f"{output_prefix}{video_title}"))
            if audio_file:
                txt_file = transcribe_audio_to_txt(audio_file, include_timestamps=include_timestamps, language="zh", model_size=model_size)
                if txt_file:
                    subtitle_files.append({
                        'language': 'zh',
                        'filename': os.path.basename(txt_file),
                        'path': f'/static/subtitles/{os.path.basename(txt_file)}',
                        'type': 'transcribed'
                    })
                else:
                    logger.error("转录失败，未生成 TXT 文件")
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    logger.info(f"已清理临时音频文件: {audio_file}")
            else:
                logger.error("音频下载失败")
        
        if subtitle_files or video_file:
            response = {
                'status': 'success',
                'message': f"'{video_title}' 的字幕和视频处理完成",
                'video_title': video_title,
                'subtitles': subtitle_files
            }
            if video_info:
                response['video'] = video_info
            return jsonify(response)
        else:
            return jsonify({
                'status': 'error',
                'message': "字幕和视频处理均失败"
            }), 500
    
    except Exception as e:
        logger.error(f"处理字幕和视频时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': f"发生错误: {str(e)}"
        }), 500

@app.route('/download_video', methods=['POST'])
def download_video_endpoint():
    """API 端点：仅下载 YouTube 视频"""
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    
    if not youtube_url:
        return jsonify({'status': 'error', 'message': '未提供 YouTube URL'}), 400
    
    unique_id = str(uuid.uuid4())[:8]
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_prefix = f"{date_string}_{unique_id}_"
    
    try:
        with yt_dlp.YoutubeDL({'skip_download': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'video')
        
        video_file = download_video(youtube_url, os.path.join(VIDEO_DIR, f"{output_prefix}{video_title}"))
        if video_file:
            return jsonify({
                'status': 'success',
                'message': f"'{video_title}' 的视频下载成功",
                'video_title': video_title,
                'video': {
                    'filename': os.path.basename(video_file),
                    'path': f'/static/videos/{os.path.basename(video_file)}'
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': "视频下载失败"
            }), 500
    except Exception as e:
        logger.error(f"处理视频时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': f"发生错误: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)