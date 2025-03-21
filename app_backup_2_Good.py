from flask import Flask, request, render_template, jsonify, send_file
import yt_dlp
import os
from datetime import datetime
import uuid
import webvtt
import logging

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建字幕存储目录
SUBTITLE_DIR = "static/subtitles"
os.makedirs(SUBTITLE_DIR, exist_ok=True)

def convert_vtt_to_txt(vtt_file_path, include_timestamps=False):
    """将 .vtt 字幕文件转换为 .txt 格式，可选择是否包含时间戳，并去重"""
    txt_file_path = vtt_file_path.replace('.vtt', '.txt')
    
    try:
        vtt = webvtt.read(vtt_file_path)
        deduplicated_captions = []
        last_text = None
        last_end = None
        
        # 将时间字符串转换为秒，便于比较
        def time_to_seconds(time_str):
            h, m, s = map(float, time_str.replace(',', '.').split(':'))
            return h * 3600 + m * 60 + s
        
        # 去重逻辑
        for caption in vtt:
            current_text = caption.text.strip()
            current_start = time_to_seconds(caption.start)
            current_end = time_to_seconds(caption.end)
            
            if last_text == current_text and last_end and abs(current_start - last_end) < 1.0:
                # 如果文本相同且时间接近，合并
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
        
        # 写入 TXT 文件
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            for caption in deduplicated_captions:
                if include_timestamps:
                    start_time = caption['start'].replace('.', ',')
                    end_time = caption['end'].replace('.', ',')
                    txt_file.write(f"{start_time} --> {end_time}: {caption['text']}\n")
                else:
                    txt_file.write(f"{caption['text']}\n")
        
        return txt_file_path
    except Exception as e:
        logger.error(f"转换 VTT 到 TXT 时出错: {e}")
        return None

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/download_subtitle', methods=['POST'])
def download_subtitle():
    """API 端点：下载 YouTube 字幕"""
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    include_timestamps = data.get('include_timestamps', False)  # 默认不包含时间戳
    
    if not youtube_url:
        return jsonify({'status': 'error', 'message': '未提供 YouTube URL'}), 400
    
    # 生成唯一标识符
    unique_id = str(uuid.uuid4())[:8]
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_prefix = f"video_{date_string}_{unique_id}_"
    
    try:
        with yt_dlp.YoutubeDL({'skip_download': True}) as ydl:
            # 提取视频信息
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'video')
            subtitles = info.get('subtitles', {})  # 手动字幕
            auto_subtitles = info.get('automatic_captions', {})  # 自动字幕
            
            # 决定使用手动还是自动字幕
            use_auto_subtitles = False
            if subtitles:
                logger.info(f"找到 '{video_title}' 的手动字幕")
            elif auto_subtitles:
                logger.info(f"'{video_title}' 无手动字幕，使用自动字幕")
                subtitles = auto_subtitles
                use_auto_subtitles = True
            else:
                return jsonify({
                    'status': 'error',
                    'message': f"视频 '{video_title}' 没有任何字幕（手动或自动生成）"
                }), 404
            
            # 配置 yt-dlp 下载选项
            ydl_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': use_auto_subtitles,
                'subtitlesformat': 'vtt',
                'outtmpl': os.path.join(SUBTITLE_DIR, f'{output_prefix}%(title)s.%(ext)s'),
            }
            
            # 下载字幕
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([youtube_url])
            
            # 收集下载的字幕文件信息
            subtitle_files = []
            for lang, sub_info in subtitles.items():
                subtitle_file = f"{output_prefix}{video_title}.{lang}.vtt"
                full_path = os.path.join(SUBTITLE_DIR, subtitle_file)
                
                if os.path.exists(full_path):
                    # 转换为 TXT，传递时间戳选项
                    txt_file = convert_vtt_to_txt(full_path, include_timestamps=include_timestamps)
                    if txt_file:
                        subtitle_files.append({
                            'language': lang,
                            'filename': os.path.basename(txt_file),
                            'path': f'/static/subtitles/{os.path.basename(txt_file)}',
                            'type': 'auto-generated' if use_auto_subtitles else 'manual'
                        })
            
            if subtitle_files:
                return jsonify({
                    'status': 'success',
                    'message': f"'{video_title}' 的字幕下载成功",
                    'video_title': video_title,
                    'subtitles': subtitle_files
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': "字幕下载失败"
                }), 500
    
    except Exception as e:
        logger.error(f"下载字幕时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': f"发生错误: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)