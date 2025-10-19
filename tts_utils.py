"""
TTS 工具模块 - Qwen TTS 官方实现
严格按照 qwen3-tts-flash 官方示例
"""

import os
import base64
import uuid


def speak_with_qwen(text, voice="Marcus", model="qwen3-tts-flash"):
    """
    使用 Qwen TTS - 按照官方注释的正确接口
    官方注释：dashscope.audio.qwen_tts.SpeechSynthesizer.call(...)
    
    Args:
        text: 要转换的文本
        voice: 音色（Marcus 或 Ethan）
        model: TTS 模型（默认 qwen3-tts-flash）
    
    Returns:
        tuple: (success, audio_data_base64 or error_message)
    """
    try:
        import requests
        from dashscope.audio.qwen_tts import SpeechSynthesizer
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return False, "Missing API Key"
        
        print(f"[TTS DEBUG] Model: {model}, Voice: {voice}")
        
        # 按照官方注释：dashscope.audio.qwen_tts.SpeechSynthesizer.call(...)
        response = SpeechSynthesizer.call(
            model=model,
            api_key=api_key,
            text=text,
            voice=voice,
            format='mp3'
        )
        
        print(f"[TTS DEBUG] Response type: {type(response)}")
        
        # 响应是 dict-like 对象，用字典方式访问
        audio_url = None
        
        if hasattr(response, 'output'):
            print(f"[TTS DEBUG] Has output attribute")
            output = response.output
            print(f"[TTS DEBUG] output type: {type(output)}")
            print(f"[TTS DEBUG] output: {output}")
            
            if hasattr(output, 'audio'):
                print(f"[TTS DEBUG] Has audio attribute")
                audio = output.audio
                print(f"[TTS DEBUG] audio type: {type(audio)}")
                print(f"[TTS DEBUG] audio: {audio}")
                
                # audio 可能是 dict 或对象
                if isinstance(audio, dict):
                    audio_url = audio.get('url')
                    print(f"[TTS DEBUG] Extracted URL (dict): {audio_url}")
                else:
                    audio_url = getattr(audio, 'url', None)
                    print(f"[TTS DEBUG] Extracted URL (attr): {audio_url}")
            elif isinstance(output, dict) and 'audio' in output:
                print(f"[TTS DEBUG] output is dict, extracting audio")
                audio = output['audio']
                print(f"[TTS DEBUG] audio from dict: {audio}")
                audio_url = audio.get('url') if isinstance(audio, dict) else getattr(audio, 'url', None)
                print(f"[TTS DEBUG] Extracted URL (from dict): {audio_url}")
        
        if audio_url:
            print(f"[TTS DEBUG] Audio URL: {audio_url}")
            
            # 下载音频
            audio_response = requests.get(audio_url, timeout=10)
            audio_response.raise_for_status()
            
            # 转 base64
            audio_data = audio_response.content
            b64_audio = base64.b64encode(audio_data).decode()
            
            print(f"[TTS DEBUG] ✅ Success! Audio size: {len(audio_data)} bytes")
            return True, b64_audio
        
        return False, f"No audio URL in response: {response}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Qwen TTS failed: {str(e)}"


def speak(text, voice="Marcus", timeout=10):
    """
    Qwen TTS 语音合成函数
    
    Args:
        text: 要转换的文本
        voice: 音色（Marcus 或 Ethan）
        timeout: 超时时间（秒）
    
    Returns:
        tuple: (success, audio_html or error_message, method_used)
    """
    
    # 使用 Qwen TTS（qwen3-tts-flash）
    print(f"[TTS] Calling Qwen TTS (voice: {voice})...")
    success, result = speak_with_qwen(text, voice=voice, model="qwen3-tts-flash")
    
    if success:
        print(f"[TTS] ✅ Qwen TTS succeeded")
        audio_id = str(uuid.uuid4())
        audio_html = f"""
            <audio id="{audio_id}" autoplay>
                <source src="data:audio/mp3;base64,{result}" type="audio/mp3">
            </audio>
            <script>
                const audio = document.getElementById('{audio_id}');
                if (audio) {{
                    audio.play().catch(e => console.log('Playback error:', e));
                }}
            </script>
        """
        return True, audio_html, "Qwen TTS"
    else:
        # TTS 失败
        print(f"[TTS] ❌ Qwen TTS failed: {result}")
        return False, f"TTS failed: {result}", "None"


def cleanup_audio_files():
    """清理旧的音频文件（预留函数，Qwen TTS 不需要本地文件）"""
    pass
