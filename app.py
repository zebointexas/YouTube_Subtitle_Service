from flask import Flask, request, render_template, jsonify, send_file
import yt_dlp
import os
from datetime import datetime
import uuid
import webvtt
import whisper
import logging
import threading
import time
import librosa  # 新增依赖，用于音频长度估计

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

# 下载状态跟踪器
download_status = {}  # 格式: {job_id: {'video_status': 'pending/complete', 'subtitle_status': 'pending/complete', 'audio_status': 'pending/complete', 'video_file': None, 'audio_file': None}}

# 转录进度跟踪器
transcription_progress = {}  # 格式: {job_id: {'progress': 0-100, 'started_at': timestamp, 'estimated_total_time': seconds, 'status': 'running/complete/failed'}}

def convert_vtt_to_txt(vtt_file_path, include_timestamps=False):
    """将 .vtt 字幕文件转换为 .txt 格式，可选择是否包含时间戳，并去重"""
    txt_file_path = vtt_file_path.replace('.vtt', '.txt')
    
    if not os.path.exists(vtt_file_path):
        logger.error(f"VTT 文件不存在: {vtt_file_path}")
        return None
    
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
        
        if os.path.exists(txt_file_path):
            logger.info(f"字幕转换成功: {txt_file_path}")
            return txt_file_path
        else:
            logger.error(f"TXT 文件未能生成: {txt_file_path}")
            return None
    except Exception as e:
        logger.error(f"转换 VTT 到 TXT 时出错: {e}")
        return None

def download_audio(youtube_url, output_file, job_id=None):
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
                            if job_id and job_id in download_status:
                                download_status[job_id]['audio_status'] = 'complete'
                                download_status[job_id]['audio_file'] = mp3_file
                            return mp3_file
                        else:
                            logger.error(f"MP3 转换失败: {mp3_file}")
                            if job_id and job_id in download_status:
                                download_status[job_id]['audio_status'] = 'failed'
                            return None
                    if job_id and job_id in download_status:
                        download_status[job_id]['audio_status'] = 'complete'
                        download_status[job_id]['audio_file'] = audio_file
                    return audio_file
            logger.error(f"未找到任何音频文件: {output_file}")
            if job_id and job_id in download_status:
                download_status[job_id]['audio_status'] = 'failed'
            return None
    except Exception as e:
        logger.error(f"下载音频时出错: {e}")
        if job_id and job_id in download_status:
            download_status[job_id]['audio_status'] = 'failed'
        return None

def async_download_audio(youtube_url, output_file, job_id):
    """异步下载音频，更新下载状态"""
    if job_id not in download_status:
        download_status[job_id] = {
            'video_status': 'pending',
            'subtitle_status': 'pending',
            'audio_status': 'pending',
            'video_file': None,
            'audio_file': None,
            'output_file': output_file
        }
    else:
        download_status[job_id]['audio_status'] = 'pending'
    
    def download_task():
        try:
            audio_file = download_audio(youtube_url, output_file, job_id)
            if audio_file:
                download_status[job_id]['audio_status'] = 'complete'
                download_status[job_id]['audio_file'] = audio_file
            else:
                download_status[job_id]['audio_status'] = 'failed'
        except Exception as e:
            logger.error(f"异步音频下载任务出错 ({job_id}): {e}")
            download_status[job_id]['audio_status'] = 'failed'
    
    thread = threading.Thread(target=download_task)
    thread.daemon = True
    thread.start()
    
    return job_id

def download_video(youtube_url, output_file, job_id=None):
    """下载 YouTube 视频，可选异步追踪下载状态"""
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
                if job_id and job_id in download_status:
                    download_status[job_id]['video_status'] = 'complete'
                    download_status[job_id]['video_file'] = video_file
                return video_file
            logger.error(f"视频文件未生成: {video_file}")
            return None
    except Exception as e:
        logger.error(f"下载视频时出错: {e}")
        return None

def async_download_video(youtube_url, output_file, job_id):
    """异步下载视频，更新下载状态"""
    if job_id not in download_status:
        download_status[job_id] = {
            'video_status': 'pending',
            'subtitle_status': 'pending',
            'audio_status': 'pending',
            'video_file': None,
            'audio_file': None,
            'output_file': output_file
        }
    else:
        download_status[job_id]['video_status'] = 'pending'
    
    def download_task():
        try:
            video_file = download_video(youtube_url, output_file)
            if video_file:
                download_status[job_id]['video_status'] = 'complete'
                download_status[job_id]['video_file'] = video_file
            else:
                download_status[job_id]['video_status'] = 'failed'
        except Exception as e:
            logger.error(f"异步视频下载任务出错 ({job_id}): {e}")
            download_status[job_id]['video_status'] = 'failed'
    
    thread = threading.Thread(target=download_task)
    thread.daemon = True
    thread.start()
    
    return job_id

def transcribe_audio_to_txt(audio_file, include_timestamps=False, language="zh", model_size="small", job_id=None):
    """使用 Whisper 将音频转换为 TXT 格式，支持不同模型，并跟踪进度"""
    txt_file_path = os.path.join(SUBTITLE_DIR, os.path.basename(audio_file).replace('.mp3', '.txt'))
    
    try:
        if not os.path.exists(audio_file):
            logger.error(f"音频文件不存在: {audio_file}")
            if job_id and job_id in transcription_progress:
                transcription_progress[job_id]['status'] = 'failed'
            return None
        
        # 初始化进度跟踪
        if job_id:
            transcription_progress[job_id] = {
                'progress': 0,
                'started_at': time.time(),
                'estimated_total_time': None,
                'status': 'running'
            }
        
        # 估算音频长度，用于计算进度
        audio_duration = None
        try:
            audio_duration = librosa.get_duration(path=audio_file)
            logger.info(f"音频长度估计: {audio_duration} 秒")
            
            time_factors = {
                'base': 0.1,
                'small': 0.3,
                'medium': 0.8,
            }
            
            if job_id and audio_duration:
                factor = time_factors.get(model_size, 0.3)
                estimated_time = audio_duration * factor + 5
                transcription_progress[job_id]['estimated_total_time'] = estimated_time
                logger.info(f"估计转录总时间: {estimated_time} 秒")
        except Exception as e:
            logger.warning(f"无法估计音频长度: {e}")
        
        if job_id:
            transcription_progress[job_id]['progress'] = 10
        
        logger.info(f"加载 Whisper 模型 ({model_size})，语言: {language}")
        model = whisper.load_model(model_size)
        
        if job_id:
            transcription_progress[job_id]['progress'] = 20
        
        logger.info(f"开始转录音频: {audio_file}")
        
        def transcribe_with_progress(audio_path):
            result = model.transcribe(audio_path, language=language, fp16=False)
            if job_id:
                transcription_progress[job_id]['progress'] = 90
            return result
        
        result = transcribe_with_progress(audio_file)
        
        if job_id:
            transcription_progress[job_id]['progress'] = 95
        
        if not result["segments"]:
            logger.error("转录结果为空")
            if job_id and job_id in transcription_progress:
                transcription_progress[job_id]['status'] = 'failed'
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
        
        if job_id and job_id in transcription_progress:
            transcription_progress[job_id]['progress'] = 100
            transcription_progress[job_id]['status'] = 'complete'
        
        logger.info(f"转录完成，生成文件: {txt_file_path}")
        return txt_file_path
    except Exception as e:
        logger.error(f"转录音频到 TXT 时出错: {e}")
        if job_id and job_id in transcription_progress:
            transcription_progress[job_id]['status'] = 'failed'
        return None

def async_transcribe_audio(audio_file, include_timestamps, language, model_size, job_id):
    """异步转录音频，更新转录状态"""
    
    def transcribe_task():
        try:
            txt_file = transcribe_audio_to_txt(audio_file, include_timestamps=include_timestamps, 
                                            language=language, model_size=model_size, job_id=job_id)
            if job_id in download_status:
                if txt_file:
                    download_status[job_id]['subtitle_status'] = 'complete'
                else:
                    download_status[job_id]['subtitle_status'] = 'failed'
        except Exception as e:
            logger.error(f"异步转录任务出错 ({job_id}): {e}")
            if job_id in download_status:
                download_status[job_id]['subtitle_status'] = 'failed'
            if job_id in transcription_progress:
                transcription_progress[job_id]['status'] = 'failed'
    
    thread = threading.Thread(target=transcribe_task)
    thread.daemon = True
    thread.start()
    
    return job_id

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/download_subtitle', methods=['POST'])
def download_subtitle():
    """API 端点：下载 YouTube 字幕或转录音频，同时下载视频和音频"""
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    include_timestamps = data.get('include_timestamps', False)
    model_size = data.get('model_size', 'small')
    
    if not youtube_url:
        return jsonify({'status': 'error', 'message': '未提供 YouTube URL'}), 400
    
    valid_models = ['base', 'small', 'medium']
    if model_size not in valid_models:
        return jsonify({'status': 'error', 'message': f"无效的模型大小: {model_size}，支持: {valid_models}"}), 400
    
    unique_id = str(uuid.uuid4())[:8]
    job_id = unique_id
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_prefix = f"{date_string}_{unique_id}_"
    
    try:
        with yt_dlp.YoutubeDL({'skip_download': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'video')
            subtitles = info.get('subtitles', {})
        
        # 异步下载视频和音频
        video_output_path = os.path.join(VIDEO_DIR, f"{output_prefix}{video_title}")
        async_download_video(youtube_url, video_output_path, job_id)
        
        audio_output_path = os.path.join(AUDIO_DIR, f"{output_prefix}{video_title}")
        async_download_audio(youtube_url, audio_output_path, job_id)
        
        subtitle_files = []
        if subtitles:
            logger.info(f"找到 '{video_title}' 的手动字幕")
            ydl_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': False,
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
                    else:
                        logger.error(f"字幕转换失败: {full_path}")
                        subtitle_files.append({
                            'language': lang,
                            'status': 'failed',
                            'type': 'manual'
                        })
                else:
                    logger.error(f"字幕文件未下载: {full_path}")
        else:
            logger.info(f"'{video_title}' 无手动字幕，开始提取音频并转录为中文 (模型: {model_size})")
            audio_file = None
            for _ in range(30):
                if job_id in download_status and download_status[job_id]['audio_status'] == 'complete':
                    audio_file = download_status[job_id]['audio_file']
                    break
                time.sleep(5)
            
            if not audio_file:
                audio_file = download_audio(youtube_url, audio_output_path, job_id)
            
            if audio_file:
                if job_id in download_status:
                    download_status[job_id]['subtitle_status'] = 'processing'
                async_transcribe_audio(audio_file, include_timestamps, "zh", model_size, job_id)
                subtitle_files.append({
                    'language': 'zh',
                    'status': 'processing',
                    'type': 'transcribing',
                    'job_id': job_id
                })
            else:
                logger.error("音频下载失败")
                if job_id in download_status:
                    download_status[job_id]['subtitle_status'] = 'failed'
        
        video_ready = False
        audio_ready = False
        video_info = None
        audio_info = None
        
        if job_id in download_status:
            if download_status[job_id]['video_status'] == 'complete':
                video_ready = True
                video_file = download_status[job_id]['video_file']
                if video_file:
                    video_info = {
                        'filename': os.path.basename(video_file),
                        'path': f'/static/videos/{os.path.basename(video_file)}'
                    }
            if download_status[job_id]['audio_status'] == 'complete':
                audio_ready = True
                audio_file = download_status[job_id]['audio_file']
                if audio_file:
                    audio_info = {
                        'filename': os.path.basename(audio_file),
                        'path': f'/static/audio/{os.path.basename(audio_file)}'
                    }
        
        response = {
            'status': 'success',
            'message': f"'{video_title}' 的处理已开始",
            'video_title': video_title,
            'subtitles': subtitle_files,
            'job_id': job_id,
            'video_ready': video_ready,
            'audio_ready': audio_ready,
            'needs_transcription_polling': not subtitles and audio_file is not None
        }
        
        if video_info:
            response['video'] = video_info
        if audio_info:
            response['audio'] = audio_info
            
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"处理字幕、视频和音频时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': f"发生错误: {str(e)}",
            'job_id': job_id
        }), 500

@app.route('/check_download_status/<job_id>', methods=['GET'])
def check_download_status(job_id):
    """检查视频和音频下载状态"""
    if job_id not in download_status:
        return jsonify({
            'status': 'error',
            'message': f"找不到下载任务: {job_id}"
        }), 404
    
    job_data = download_status[job_id]
    video_ready = job_data['video_status'] == 'complete'
    audio_ready = job_data['audio_status'] == 'complete'
    
    response = {
        'status': 'success',
        'video_status': job_data['video_status'],
        'audio_status': job_data['audio_status'],
        'subtitle_status': job_data['subtitle_status'],
        'video_ready': video_ready,
        'audio_ready': audio_ready
    }
    
    if video_ready and job_data['video_file']:
        response['video'] = {
            'filename': os.path.basename(job_data['video_file']),
            'path': f'/static/videos/{os.path.basename(job_data["video_file"])}'
        }
    if audio_ready and job_data['audio_file']:
        response['audio'] = {
            'filename': os.path.basename(job_data['audio_file']),
            'path': f'/static/audio/{os.path.basename(job_data["audio_file"])}'
        }
    
    return jsonify(response)

@app.route('/check_transcription_progress/<job_id>', methods=['GET'])
def check_transcription_progress(job_id):
    """获取音频转录进度"""
    if job_id not in transcription_progress:
        return jsonify({
            'status': 'error',
            'message': f"找不到转录任务: {job_id}"
        }), 404
    
    progress_data = transcription_progress[job_id]
    current_time = time.time()
    elapsed_time = current_time - progress_data['started_at']
    
    remaining_time = None
    if progress_data['progress'] > 0 and progress_data['estimated_total_time']:
        progress_ratio = progress_data['progress'] / 100.0
        if progress_ratio > 0:
            adjusted_total_time = elapsed_time / progress_ratio
            remaining_time = max(0, adjusted_total_time - elapsed_time)
    
    response = {
        'status': progress_data['status'],
        'progress': progress_data['progress'],
        'elapsed_time': round(elapsed_time, 2),
        'estimated_total_time': round(progress_data['estimated_total_time'], 2) if progress_data['estimated_total_time'] else None,
        'remaining_time': round(remaining_time, 2) if remaining_time else None
    }
    
    return jsonify(response)

@app.route('/get_transcription_result/<job_id>', methods=['GET'])
def get_transcription_result(job_id):
    """获取已完成转录任务的结果"""
    if job_id not in download_status:
        return jsonify({
            'status': 'error',
            'message': f"找不到任务: {job_id}"
        }), 404
    
    if job_id not in transcription_progress:
        return jsonify({
            'status': 'error',
            'message': f"找不到转录任务: {job_id}"
        }), 404
    
    progress_data = transcription_progress[job_id]
    
    if progress_data['status'] != 'complete':
        return jsonify({
            'status': progress_data['status'],
            'progress': progress_data['progress'],
            'message': "转录尚未完成"
        })
    
    if job_id in download_status and download_status[job_id]['audio_file']:
        audio_file = download_status[job_id]['audio_file']
        txt_file_name = os.path.basename(audio_file).replace('.mp3', '.txt')
        txt_file_path = os.path.join(SUBTITLE_DIR, txt_file_name)
        
        if os.path.exists(txt_file_path):
            return jsonify({
                'status': 'success',
                'subtitle': {
                    'language': 'zh',
                    'filename': txt_file_name,
                    'path': f'/static/subtitles/{txt_file_name}',
                    'type': 'transcribed'
                }
            })
    
    return jsonify({
        'status': 'error',
        'message': "转录已完成但找不到字幕文件"
    }), 500

@app.route('/download_video', methods=['POST'])
def download_video_endpoint():
    """API 端点：下载 YouTube 视频和音频"""
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    
    if not youtube_url:
        return jsonify({'status': 'error', 'message': '未提供 YouTube URL'}), 400
    
    unique_id = str(uuid.uuid4())[:8]
    job_id = unique_id
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_prefix = f"{date_string}_{unique_id}_"
    
    try:
        with yt_dlp.YoutubeDL({'skip_download': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'video')
        
        video_output_path = os.path.join(VIDEO_DIR, f"{output_prefix}{video_title}")
        async_download_video(youtube_url, video_output_path, job_id)
        
        audio_output_path = os.path.join(AUDIO_DIR, f"{output_prefix}{video_title}")
        async_download_audio(youtube_url, audio_output_path, job_id)
        
        return jsonify({
            'status': 'success',
            'message': f"'{video_title}' 的视频和音频下载已开始",
            'video_title': video_title,
            'job_id': job_id
        })
    except Exception as e:
        logger.error(f"处理视频和音频时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': f"发生错误: {str(e)}"
        }), 500

@app.route('/download_audio', methods=['POST'])
def download_audio_endpoint():
    """API 端点：仅下载 YouTube 音频"""
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    
    if not youtube_url:
        return jsonify({'status': 'error', 'message': '未提供 YouTube URL'}), 400
    
    unique_id = str(uuid.uuid4())[:8]
    job_id = unique_id
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_prefix = f"{date_string}_{unique_id}_"
    
    try:
        with yt_dlp.YoutubeDL({'skip_download': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'video')
        
        audio_output_path = os.path.join(AUDIO_DIR, f"{output_prefix}{video_title}")
        async_download_audio(youtube_url, audio_output_path, job_id)
        
        return jsonify({
            'status': 'success',
            'message': f"'{video_title}' 的音频下载已开始",
            'video_title': video_title,
            'job_id': job_id
        })
    except Exception as e:
        logger.error(f"处理音频时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': f"发生错误: {str(e)}"
        }), 500

def clean_expired_download_status():
    """定期清理超过1小时的下载状态记录"""
    while True:
        now = time.time()
        expired_jobs = []
        
        for job_id, job_data in download_status.items():
            if 'created_at' not in job_data:
                job_data['created_at'] = now
            if now - job_data['created_at'] > 3600:
                expired_jobs.append(job_id)
        
        for job_id in expired_jobs:
            del download_status[job_id]
            logger.info(f"清理过期下载任务: {job_id}")
        
        expired_transcriptions = []
        for job_id, progress_data in transcription_progress.items():
            if 'started_at' not in progress_data:
                progress_data['started_at'] = now
            if now - progress_data['started_at'] > 3600:
                expired_transcriptions.append(job_id)
        
        for job_id in expired_transcriptions:
            del transcription_progress[job_id]
            logger.info(f"清理过期转录任务: {job_id}")
        
        time.sleep(600)

cleanup_thread = threading.Thread(target=clean_expired_download_status)
cleanup_thread.daemon = True
cleanup_thread.start()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)