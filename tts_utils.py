"""
TTS 工具模块 - Qwen TTS 官方实现 + European Portuguese Support
严格按照 qwen3-tts-flash 官方示例
"""

import os
import base64
import uuid
import tempfile
import streamlit.components.v1 as components


def speak_with_qwen(text, voice="Cherry", model="qwen3-tts-flash"):
    """
    使用 Qwen TTS - 按照官方注释的正确接口
    官方注释：dashscope.audio.qwen_tts.SpeechSynthesizer.call(...)
    
    Args:
        text: 要转换的文本
        voice: 音色（Cherry 或 Ethan）
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


def speak_with_openai_european_portuguese(text, voice="onyx"):
    """
    使用 OpenAI TTS 生成欧洲葡萄牙语语音
    """
    try:
        from openai import OpenAI
        import base64
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "Missing OpenAI API Key"
        
        client = OpenAI(api_key=api_key)
        
        print(f"[OpenAI TTS] Generating European Portuguese audio with voice: {voice}")
        
        # 使用 OpenAI TTS - 虽然没有直接的葡萄牙语变体参数，但使用合适的音色
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,  # onyx 或 alloy 对葡萄牙语效果较好
            input=text
        )
        
        # Convert audio to base64 instead of saving to file
        audio_data = response.content
        b64_audio = base64.b64encode(audio_data).decode()
        
        print(f"[OpenAI TTS] ✅ Success! European Portuguese audio size: {len(audio_data)} bytes")
        return True, b64_audio
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"OpenAI European Portuguese TTS failed: {str(e)}"


def speak_with_azure_european_portuguese(text, voice="pt-PT-FernandaNeural"):
    """
    使用 Azure TTS 生成高质量欧洲葡萄牙语语音（推荐）
    """
    try:
        import requests
        
        azure_key = os.getenv("AZURE_TTS_KEY")
        azure_region = os.getenv("AZURE_TTS_REGION", "westeurope")
        
        if not azure_key:
            return False, "Azure TTS key not found"
        
        print(f"[Azure TTS] Generating European Portuguese audio with voice: {voice}")
        
        # Azure TTS request
        headers = {
            'Ocp-Apim-Subscription-Key': azure_key,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3'
        }
        
        ssml = f"""
        <speak version='1.0' xml:lang='pt-PT'>
            <voice xml:lang='pt-PT' name='{voice}'>
                {text}
            </voice>
        </speak>
        """
        
        response = requests.post(
            f"https://{azure_region}.tts.speech.microsoft.com/cognitiveservices/v1",
            headers=headers,
            data=ssml.encode('utf-8'),
            timeout=10
        )
        
        if response.status_code == 200:
            # Convert to base64
            audio_data = response.content
            b64_audio = base64.b64encode(audio_data).decode()
            
            print(f"[Azure TTS] ✅ Success! European Portuguese audio size: {len(audio_data)} bytes")
            return True, b64_audio
        else:
            return False, f"Azure TTS failed: {response.status_code} - {response.text}"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Azure TTS error: {str(e)}"


def speak(text, voice="Cherry", timeout=10, language="English", portuguese_variant="european"):
    """
    智能 TTS 函数：英语用 Qwen TTS，葡萄牙语用 European Portuguese TTS
    """
    
    if language == "Portuguese":
        # 优先尝试 Azure TTS（最佳质量）
        if portuguese_variant == "european":
            print(f"[TTS] Using Azure TTS for European Portuguese...")
            success, result = speak_with_azure_european_portuguese(text, voice="pt-PT-FernandaNeural")
            
            if success:
                print(f"[TTS] ✅ Azure European Portuguese TTS succeeded")
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
                return True, audio_html, "Azure European Portuguese TTS"
            
            # Azure 失败，降级到 OpenAI
            print(f"[TTS] ❌ Azure TTS failed, falling back to OpenAI: {result}")
        
        # 使用 OpenAI TTS 生成欧洲葡萄牙语语音
        print(f"[TTS] Using OpenAI TTS for European Portuguese (voice: onyx)...")
        success, result = speak_with_openai_european_portuguese(text, voice="onyx")
        
        if success:
            print(f"[TTS] ✅ OpenAI European Portuguese TTS succeeded")
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
            return True, audio_html, "OpenAI European Portuguese TTS"
        else:
            # OpenAI TTS 失败，降级到 gTTS with European Portuguese
            print(f"[TTS] ❌ OpenAI European Portuguese TTS failed: {result}")
            return _fallback_gtts_european_portuguese(text)
            
    else:
        # 使用 Qwen TTS 生成英语语音
        print(f"[TTS] Using Qwen TTS for English (voice: {voice})...")
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
            # Qwen TTS 失败，降级到 gTTS
            print(f"[TTS] ❌ Qwen TTS failed: {result}")
            return _fallback_gtts(text, "en")


def _fallback_gtts_european_portuguese(text):
    """
    gTTS 降级方案 - 欧洲葡萄牙语
    """
    try:
        from gtts import gTTS
        import tempfile
        
        print(f"[TTS] Falling back to gTTS for European Portuguese...")
        
        # gTTS 使用 'pt' 参数，但发音更接近欧洲葡萄牙语
        tts = gTTS(text=text, lang='pt', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            
        audio_html = f"""
            <audio autoplay controls style="width: 100%; height: 50px;">
                <source src="{tmp_file.name}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        """
        print(f"[TTS] ✅ gTTS European Portuguese fallback succeeded")
        return True, audio_html, "gTTS European Portuguese (fallback)"
        
    except Exception as e:
        error_msg = f"All European Portuguese TTS methods failed. Last error: {str(e)}"
        print(f"[TTS] ❌ {error_msg}")
        return False, error_msg, "None"


def _fallback_gtts(text, lang="en"):
    """
    gTTS 降级方案
    """
    try:
        from gtts import gTTS
        import tempfile
        
        print(f"[TTS] Falling back to gTTS (lang: {lang})...")
        
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            
        audio_html = f"""
            <audio autoplay controls style="width: 100%; height: 50px;">
                <source src="{tmp_file.name}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        """
        print(f"[TTS] ✅ gTTS fallback succeeded")
        return True, audio_html, "gTTS (fallback)"
        
    except Exception as e:
        error_msg = f"All TTS methods failed. Last error: {str(e)}"
        print(f"[TTS] ❌ {error_msg}")
        return False, error_msg, "None"


def cleanup_audio_files():
    """清理临时音频文件"""
    try:
        # 清理可能遗留的临时文件
        import glob
        temp_files = glob.glob("/tmp/tmp*mp3") + glob.glob("/var/folders/*/*/tmp*mp3")
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        print(f"[TTS Cleanup] Cleaned {len(temp_files)} temporary files")
    except Exception as e:
        print(f"[TTS Cleanup] Error during cleanup: {e}")
