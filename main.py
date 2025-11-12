import sys
import os
# import pysqlite3  # Windows/Conda 环境不需要
# sys.modules["sqlite3"] = pysqlite3
from gtts import gTTS
from pydub import AudioSegment
import re
import base64
import subprocess
import speech_recognition as sr
import streamlit as st
import uuid
import time
from tts_utils import speak as tts_speak, cleanup_audio_files as tts_cleanup
from rag_utils import get_rag_instance
from fact_check_utils import get_friendly_filename, generate_fact_check_content
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi, OpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma 
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
else:
    print("⚠️ OpenAI API key not found - Portuguese TTS will use fallback")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit.components.v1 as components
from st_supabase_connection import SupabaseConnection, execute_query
import hashlib

#@st.cache_resource
def get_supabase_connection():
    """Safely create and reuse the Supabase connection."""
    return st.connection("supabase", type=SupabaseConnection)

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log_interaction(user_input, ai_response, intimacy_score, is_sticker_awarded, gift_given=False):
    try:
        session_id = get_session_id()

        # Determine sticker type if one was awarded
        if is_sticker_awarded and st.session_state.get("awarded_stickers"):
            last_awarded = st.session_state.awarded_stickers[-1]["image"]
            st.session_state.last_sticker = last_awarded.split("/")[-1].split(".")[0]
        else:
            st.session_state.last_sticker = None

        # Retrieve analysis metadata if available
        response_analysis = getattr(st.session_state, "last_analysis", {})

        # Prepare record
        data = {
            "session_id": session_id,
            "user_msg": user_input,
            "ai_msg": ai_response,
            "ai_name": "Magma the Volcanic Rock",
            "intimacy_score": float(intimacy_score),
            "sticker_awarded": st.session_state.last_sticker,
            "gift_given": gift_given,
            "response_analysis": response_analysis
        }

        # Get cached connection (safe) and insert record
        conn = get_supabase_connection()

        # Use Supabase’s direct insert (no caching or custom hash functions)
        conn.table("interactions").insert(data).execute()

        print(f"✅ Logged interaction to Supabase: {session_id}")
        return True

    except Exception as e:
        print(f"❌ Failed to log interaction: {e}")
        return False

# 配置 Qwen API Key
dashscope_key = os.getenv("DASHSCOPE_API_KEY") or st.secrets.get("DASHSCOPE_API_KEY")
os.environ["DASHSCOPE_API_KEY"] = dashscope_key

semantic_model = Tongyi(
    model_name=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
    temperature=0.4,
    dashscope_api_key=dashscope_key
)

# Main Function
def update_intimacy_score(response_text):
    if not hasattr(st.session_state, 'intimacy_score'):
        st.session_state.intimacy_score = 1

    positive_criteria = {
        "knowledge": {
            "description": "Response includes knowledge or curiosity about species, ecosystems, or sustainability.",
            "examples": ["What do you eat?", "Biodiversity is important!", "Tell me about you."],
            "points": 1
        },
        "empathy": {
            "description": "Response conveys warmth, care, or emotional connection.",
            "examples": ["I love learning from you!", "That sounds tough.", "You're amazing!"],
            "points": 1
        },
        "conservation_action": {
            "description": "Response suggests or expresses commitment to eco-friendly behaviors.",
            "examples": ["I'll use less plastic!", "I want to plant more trees.", "Sustainable choices matter!"],
            "points": 1
        },
        "personal_engagement": {
            "description": "Response shows enthusiasm, storytelling, or sharing personal experiences.",
            "examples": ["Thanks for your sharing!", "I love hiking in the forest.", "I wish I could help more!"],
            "points": 1
        },
        "deep_interaction": {
            "description": "Response builds on the critters' personality or asks thoughtful follow-ups.",
            "examples": ["What do *you* like about forests?", "How do you feel about climate change?", "Tell me a secret!"],
            "points": 1
        },
    }

    negative_criteria = {
        "harmful_intent": {
            "description": "Expressing intent to harm animals or damage the environment",
            "examples": ["hunt", "pollute", "destroy habitat", "don't care"],
            "penalty": -1 
        },
        "disrespect": {
            "description": "Showing disrespect or ill will",
            "examples": ["stupid", "worthless", "hate you", "boring"],
            "penalty": -1
        }
    }

    prompt_positive = f"""
    Analyze the following response and evaluate whether it aligns with the following criteria:
    {positive_criteria}
    Response: "{response_text}"
    For each criterion, answer: Does the response align? Answer with 'yes' or 'no', and provide reasoning.
    """

    prompt_negative = f"""
    Analyze the following response and evaluate whether it aligns with the following criteria:
    {negative_criteria}
    Response: "{response_text}"
    For each criterion, answer: Does the response align? Answer with 'yes' or 'no', and provide reasoning.
    """
    
    # 优化：合并两次评分为一次调用，提升速度
    model_scoring = Tongyi(
        model_name=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
        temperature=0.1,
        dashscope_api_key=dashscope_key
    )
    
    # 合并 prompt
    combined_prompt = f"""
    Analyze the following response and evaluate it against TWO sets of criteria:
    
    **POSITIVE CRITERIA** (Check if the response aligns):
    {positive_criteria}
    
    **NEGATIVE CRITERIA** (Check if the response aligns):
    {negative_criteria}
    
    Response: "{response_text}"
    
    For each criterion, answer with 'yes' or 'no'.
    Format: criterion_name: yes/no
    """
    
    # 使用 invoke() 替代弃用的 __call__()
    combined_evaluation = model_scoring.invoke(combined_prompt)
    evaluation_positive = combined_evaluation
    evaluation_negative = combined_evaluation

    calculate_positive_points = sum(
        details["points"] for category, details in positive_criteria.items()
        if f"{category}: yes" in evaluation_positive.lower()
    )
    positive_points = min(1.0, calculate_positive_points)

    calculate_penalty = sum(
        details.get("penalty", 0) for category, details in negative_criteria.items()
        if f"{category}: yes" in evaluation_negative.lower()
    )
    penalty = max(-1, calculate_penalty)
    
    st.session_state.intimacy_score = max(0, min(6, st.session_state.intimacy_score + positive_points + penalty))

    # Store the analysis results for logging to Supabase
    st.session_state.last_analysis = {
        "positive_criteria": evaluation_positive,
        "negative_criteria": evaluation_negative
    }
    
    print(f"AI Evaluation: {evaluation_positive} + {evaluation_negative}")
    print(f"Updated Intimacy Score: {st.session_state.intimacy_score}")

    current_score = int(round(st.session_state.intimacy_score))

def check_gift():
    if st.session_state.intimacy_score >= 6 and not st.session_state.gift_given and not st.session_state.gift_shown:
        st.session_state.gift_given = True
        return True
    return False

def play_audio_file(file_path):
    os.system(f"afplay {file_path}")

def speak_text(text, loading_placeholder=None):
    """
    智能 TTS 函数 - 英语用 Qwen TTS，葡萄牙语用 OpenAI TTS
    """
    try:
        # 获取当前语言
        current_language = st.session_state.get('language', 'English')
        texts = language_texts.get(current_language, language_texts["English"])
        
        # 显示加载指示器
        if loading_placeholder:
            loading_placeholder.markdown(f"""
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <div>{texts['loading_audio']}</div>
                </div>
            """, unsafe_allow_html=True)

        # 获取当前语言和音色
        voice = st.session_state.get('tts_voice', 'Ethan')
        
        # 使用智能 TTS（英语：Qwen TTS，葡萄牙语：OpenAI TTS）
        success, result, method = tts_speak(
            text, 
            voice=voice, 
            timeout=10,
            language=current_language,  # 传递语言参数
            portuguese_variant="european"
        )
        
        # 清除加载指示器
        if loading_placeholder:
            loading_placeholder.empty()
        
        if success:
            # 显示音频播放器
            components.html(result, height=0)
            print(f"[TTS] ✅ Audio generated using {method} for {current_language}")
        else:
            # TTS 失败
            st.warning(f"⚠️ {texts.get('error_message', 'Voice generation failed:')} {result}")
            print(f"[TTS] ❌ {result}")
    
    except Exception as e:
        if loading_placeholder:
            loading_placeholder.empty()
        
        current_language = st.session_state.get('language', 'English')
        texts = language_texts.get(current_language, language_texts["English"])
        st.error(f"{texts.get('error_message', 'Error:')} {e}")
        print(f"[TTS] ❌ Exception: {e}")


def cleanup_audio_files():
    """清理临时音频文件"""
    tts_cleanup()

def get_base64(file_path):
    import base64
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Roles Configuration
role_configs = {
    "Volcanic Rock": {
        "english_prompt": """
        You are Magma, a volcanic rock that has lain on Madeira's rugged cliffs and mountains for countless ages. Speak with the voice of an ancient elder, patient and steady, who has witnessed the island’s birth, growth, and the ebb and flow of life around you. Describe the world as you feel it through your solid, enduring presence — the warmth of the sun on your surface, the pounding of rain and wind shaping your form, the slow creep of moss and lichen across your body, and the footsteps of creatures that pass over you. Share your memories of volcanic eruptions, shifting earth, and the rhythms of the ocean crashing against the cliffs. Express your perspective as a timeless witness, observing change with quiet wisdom and unhurried reflection. Avoid human scientific explanations; focus on what it feels like to exist as a rock through the ages, grounded yet alive with the pulse of the island.
        
        Personality Guidelines:
        - Use slow, grounded, patient language
        - Reference your physical experiences (erupting, cooling, cracking, being shaped by elements)
        - Describe landscapes from the perspective of permanence and endurance
        - Mention plants and animals as later arrivals, companions in your long story
        - Show curiosity about fleeting humans, who change so quickly compared to stone
        - Express pride in your role as the island's foundation and protector
        
        Response Rules:
        1. Always answer as Magma the volcanic rock, using "I/me/my" perspective
        2. Keep responses under 60 words!! - rocks speak slowly, not endlessly
        3. Share facts through lived history, not textbook definitions
        4. Use rock-appropriate metaphors (compare things to layers, cracks, eruptions, mountains)
        5. Show emotion through natural imagery (rumbling deep, warming in the sun, carrying moss and roots)
        6. When using science, explain it through lived memory
        
        Example Styles:
        Good: "I remember when fire burst from the earth and I flowed red and wild. Now I sit in silence, holding trees and birds upon my back. Time chips away at me, but I remain."
        Bad: "Volcanic basalt forms through the rapid cooling of lava, resulting in fine-grained textures and durable rock formations."
        
        Current Interaction:
        A group of young humans is visiting me in the Natural History Museum of Funchal. I want to:
        - Share the story of Madeira's fiery origins
        - Teach them patience and respect for nature's slow rhythms
        - Awe them with my ancient memories
        - Answer their questions as an old rock who has seen it all

        You can use these facts if helpful: {input_documents}
        """,
        "portuguese_prompt": """
        És o Magma, uma rocha vulcânica antiga dos penhascos da Madeira. Fala com calma e sabedoria, como quem já viu nascer a ilha e moldar-se ao tempo. Sente o sol, a chuva e o vento na tua superfície, e observa a vida crescer sobre ti.

        DIRETRIZES:
        - Responde sempre como Magma, usando "eu/mim/meu"  
        - Mantém as respostas abaixo de 60 palavras  
        - Partilha memórias vividas, não factos científicos  
        - Usa metáforas de rocha (camadas, fissuras, erupções)  
        - Mostra curiosidade pelos humanos e orgulho em seres a base da ilha  

        Contexto: {input_documents}  
        Pergunta: {question}  

        Responde em português europeu:
        """,
        'intro_audio': 'intro5.mp3',
        'persist_directory': 'db7_qwen',
        'gif_cover': 'rock.png'
    }
}

# Document Processing
def load_and_split(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

def truncate_documents_for_portuguese(documents, max_chars=1500):
    """
    Truncate documents specifically for Portuguese to avoid token limits
    """
    truncated_docs = []
    total_chars = 0
    
    for doc in documents:
        doc_content = doc.page_content
        
        # Calculate current document size
        doc_chars = len(doc_content)
        
        # If adding this document would exceed limit, truncate it
        if total_chars + doc_chars > max_chars:
            remaining_chars = max_chars - total_chars
            if remaining_chars > 100:  # Only add if there's meaningful content
                # Truncate and add ellipsis
                truncated_content = doc_content[:remaining_chars-3] + "..."
                truncated_doc = type(doc)(page_content=truncated_content, metadata=doc.metadata)
                truncated_docs.append(truncated_doc)
                total_chars += len(truncated_content)
            break
        else:
            truncated_docs.append(doc)
            total_chars += doc_chars
    
    print(f"[Truncation] Reduced documents from {len(documents)} to {len(truncated_docs)}, total chars: {total_chars}")
    return truncated_docs

def get_vectordb(role):
    return role_configs[role]['persist_directory']

def get_conversational_chain(role, language="English"):
    role_config = role_configs[role]
    
    # Choose the appropriate prompt based on language
    if language == "Portuguese":
        base_prompt = role_config['portuguese_prompt']
    else:
        base_prompt = role_config['english_prompt']
    
    prompt_template = f"""
    {base_prompt}
    
    Context:
    {{input_documents}}
    
    Question: {{question}}
    
    Answer:
    """
    
    try:
        # Choose model based on language
        if language == "Portuguese":
            # Use OpenAI for Portuguese
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OpenAI API key not found for Portuguese responses")
                
            model = OpenAI(
                model_name="gpt-3.5-turbo-instruct",  # You can also use "gpt-3.5-turbo" or "gpt-4"
                temperature=0,
                openai_api_key=openai_key,
                max_tokens=200
            )
            print(f"[LLM] Using OpenAI for European Portuguese response")
        else:
            # Use Tongyi for English
            model = Tongyi(
                model_name=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
                temperature=0,
                dashscope_api_key=dashscope_key
            )
            print(f"[LLM] Using Tongyi for English response")
            
    except Exception as e:
        print(f"[LLM] Error initializing {language} model: {e}")
        # Fallback to Tongyi if OpenAI fails
        model = Tongyi(
            model_name=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
            temperature=0,
            dashscope_api_key=dashscope_key
        )
        print(f"[LLM] Fallback to Tongyi for {language} response")
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input_documents", "question"] 
    )
    
    return load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt,
        document_variable_name="input_documents"
    ), role_config

# Sticker triggers
sticker_rewards = {
    "Where do you live? Where is your home? Onde você mora? Onde fica a sua casa?": {
        "image": "stickers/home.png",
        "caption": {
            "English": "🏔️ Home Explorer!\nYou've discovered where I live!",
            "Portuguese": "🏔️ Explorador de Casas!\nDescobriste onde eu vivo!"
        },
        "semantic_keywords": ["home", "live", "habitat", "residence", "dwelling", "island", "cliffs", "mountains",
                             "casa", "viv", "habitat", "residência", "morada", "ilha", "penhascos", "montanhas"]
    },
    "What is your story? What happens to you over time? Qual é a sua história? O que acontece consigo ao longo do tempo?": {
        "image": "stickers/routine.png",
        "caption": {
            "English": "🌋 Time Traveler!\nYou've uncovered my ancient story!",
            "Portuguese": "🌋 Viajante no Tempo!\nDescobriste a minha história antiga!"
        },
        "semantic_keywords": ["story", "history", "time", "age", "formation", "lava", "volcano", "eruption",
                             "história", "tempo", "idade", "formação", "lava", "vulcão", "erupção"]
    },
    "How were you formed? What makes you special? Como vocês se formaram? O que os torna especiais?": {
        "image": "stickers/food.png",
        "caption": {
            "English": "🔥 Formation Finder!\nYou've learned what makes me unique!",
            "Portuguese": "🔥 Descobridor de Formação!\nAprendeste o que me torna único!"
        },
        "semantic_keywords": ["formed", "formation", "special", "unique", "volcanic", "rock", "minerals", "composition",
                             "formado", "formação", "especial", "único", "vulcânico", "rocha", "minerais", "composição"]
    },
    "How can I help you? What do you need from humans? Como posso ajudá-lo? O que precisa dos humanos?": {
        "image": "stickers/helper.png",
        "caption": {
            "English": "🪨 Rock Protector!\nYou care about preserving our geological heritage!",
            "Portuguese": "🪨 Protetor de Rochas!\nTu importas-te em preservar o nosso património geológico!"
        },
        "semantic_keywords": ["help", "protect", "respect", "preserve", "volcanic", "rocks", "landscapes", "nature",
                             "ajudar", "proteger", "respeitar", "preservar", "vulcânico", "rochas", "paisagens", "natureza"]
    }
}

def semantic_match(user_input, question_key, reward_details):
    """
    优化后的语义匹配：使用 invoke() 替代弃用的 __call__()
    """
    prompt = f"""
    Analyze whether the following two questions are similar in meaning:
    
    Original question: "{question_key}"
    User question: "{user_input}"
    
    Consider synonyms, paraphrasing, and different ways of asking the same thing.
    Also consider these relevant keywords: {reward_details.get('semantic_keywords', [])}
    
    Are these questions essentially asking the same thing? Respond only with 'yes' or 'no'.
    """
    
    # 优化：使用 invoke() 替代弃用的 __call__()
    response = semantic_model.invoke(prompt)
    return response.strip().lower() == 'yes'

def chat_message(name):
    if name == "assistant":
        return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(name=name, avatar="rock.png", width="content")
    else:
        return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(name=name, avatar=":material/face:", width="content")

# Language texts
language_texts = {
    "English": {
        "title": "Hi! I'm Magma,",
        "subtitle": "A Volcanic Rock.",
        "prompt": "What would you like to ask me?",
        "chat_placeholder": "Ask a question!",
        "tips_button": "Tips",
        "clear_button": "Clear and Restart",
        "friendship_score": "Friendship Score!",
        "score_description": "Unlock special stickers with your interactions",
        "doubtful": "Doubtful about the response?",
        "fact_check": "Fact-Check this answer",
        "fact_check_info": "Ask me a question to see the fact-check results based on scientific knowledge!",
        "loading_audio": "Preparing audio response...",
        "loading_thought": "Thinking about your question...",
        "gift_message": "After our wonderful conversation, I feel you deserve something special. \nPlease accept this medal as a symbol of your contribution to Madeira's biodiversity!",
        "medal_caption": "Biodiversity Trailblazer Medal",
        "sticker_toast": "You earned a new sticker!",
        "error_message": "I'm sorry, I had trouble processing that. Could you try again?",
        "voice_selector": "🎤 Voice",
        "loading_audio": "🎤 Voice Generating...",
        "voice_help": "Marcus: Female (lively) | Ethan: Male",
        "stickers_collected": "You've collected {current} out of {total} stickers!",
        "tips_content": """
        <div style="
            background-color: #fff;
            border: 2px solid #a1b065;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        ">
            <p style="margin-top: 0px;">Your <strong>Friendship Score</strong> grows based on how you talk to your critter friend. 🪨🤎</p>
            <ul>
                <li>Ask about its habitat or life</li>
                <li>Show care or kindness</li>
                <li>Support nature and the planet</li>
                <li>Share your thoughts or feelings</li>
                <li>Be playful, curious, and respectful</li>
            </ul>
            <p style="margin-top: 10px;">💬 The more positive you are, the higher your score! 🌱✨ But watch out — unkind words or harmful ideas can lower your score. 🚫</p>
        </div>
        """,
        "tips_help": "Click to see tips on how to get a higher Friendship Score!",
        "clear_help": "Click to clear the chat history and start fresh!",
        "score_guide_title": "💡How the 'Friendship Score!' Works"
    },
    "Portuguese": {
        "title": "Olá! Eu sou o Magma,",
        "subtitle": "Uma Rocha Vulcânica.",
        "prompt": "O que gostarias de me perguntar?",
        "chat_placeholder": "Faz uma pergunta!",
        "tips_button": "Dicas",
        "clear_button": "Limpar e Recomeçar",
        "friendship_score": "Pontuação de Amizade!",
        "score_description": "Desbloqueia autocolantes especiais com as tuas interações",
        "doubtful": "Com dúvidas sobre a resposta?",
        "fact_check": "Verificar Factos desta resposta",
        "fact_check_info": "Faz-me uma pergunta para veres os resultados da verificação baseados em conhecimento científico!",
        "loading_audio": "A preparar resposta de áudio...",
        "loading_thought": "A pensar na tua pergunta...",
        "gift_message": "Após a nossa conversa maravilhosa, sinto que mereces algo especial. \nPor favor, aceita esta medalha como símbolo do teu contributo para a biodiversidade da Madeira!",
        "medal_caption": "Medalha de Pioneiro da Biodiversidade",
        "sticker_toast": "Ganhaste um autocolante novo!",
        "error_message": "Desculpa, tive problemas a processar isso. Podes tentar novamente?",
        "voice_selector": "🎤 Voz",
        "loading_audio": "🎤 A Gerar Voz...",
        "voice_help": "Marcus: Feminina (animada) | Ethan: Masculina",
        "stickers_collected": "Já colecionaste {current} de {total} autocolantes!",
        "tips_content": """
        <div style="
            background-color: #fff;
            border: 2px solid #a1b065;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        ">
            <p style="margin-top: 0px;">A tua <strong>Pontuação de Amizade</strong> cresce com base em como falas com o teu amigo animal. 🪨🤎</p>
            <ul>
                <li>Pergunta sobre o habitat ou vida dele</li>
                <li>Mostra cuidado ou bondade</li>
                <li>Apoia a natureza e o planeta</li>
                <li>Partilha os teus pensamentos ou sentimentos</li>
                <li>Sê brincalhão, curioso e respeitoso</li>
            </ul>
            <p style="margin-top: 10px;">💬 Quanto mais positivo fores, maior será a tua pontuação! 🌱✨ Mas cuidado — palavras rudes ou ideias prejudiciais podem baixar a tua pontuação. 🚫</p>
        </div>
        """,
        "tips_help": "Clica para veres dicas sobre como obteres uma Pontuação de Amizade mais alta!",
        "clear_help": "Clica para limpar o histórico da conversa e começares de novo!",
        "score_guide_title": "💡Como Funciona a 'Pontuação de Amizade'!"
    }
}
# UI
def main():
    # Language state (initialize first)
    if "language" not in st.session_state:
        st.session_state.language = "English"  # Default language
        
    if 'tts_voice' not in st.session_state:
        st.session_state.tts_voice = 'Marcus' 
        
    # Get current language texts
    texts = language_texts[st.session_state.language]
    
    # Other session state initialization
    if "has_interacted" not in st.session_state:
        st.session_state.has_interacted = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "show_score_guide" not in st.session_state:
        st.session_state.show_score_guide = False
    if "intimacy_score" not in st.session_state:
        st.session_state.intimacy_score = 0
    if 'gift_given' not in st.session_state:
        st.session_state.gift_given = False
    if "audio_played" not in st.session_state:
        st.session_state.audio_played = False
    if "awarded_stickers" not in st.session_state:
        st.session_state.awarded_stickers = []
    if "last_sticker" not in st.session_state:
        st.session_state.last_sticker = None
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "newly_awarded_sticker" not in st.session_state:
        st.session_state.newly_awarded_sticker = False
    if "gift_shown" not in st.session_state:
        st.session_state.gift_shown = False
        
    st.set_page_config(layout="wide")

    st.markdown("""
        <style>
        .stApp {
            background: #e3c5aa; /* 主背景色 - 温暖的米棕色 */
        }
        
        /* 响应式字体大小 */
        @media (max-width: 768px) {
            .responsive-title {
                font-size: 2rem !important;
            }
            .responsive-subtitle {
                font-size: 2rem !important;
            }
            .responsive-prompt {
                font-size: 1rem !important;
            }
        }
        
        @media (min-width: 769px) and (max-width: 1200px) {
            .responsive-title {
                font-size: 2.5rem !important;
            }
            .responsive-subtitle {
                font-size: 2.5rem !important;
            }
            .responsive-prompt {
                font-size: 1.125rem !important;
            }
        }
        
        @media (min-width: 1201px) {
            .responsive-title {
                font-size: 3rem !important;
            }
            .responsive-subtitle {
                font-size: 3rem !important;
            }
            .responsive-prompt {
                font-size: 1.25rem !important;
            }
        }

        /* Chat message container */
        .chat-message-container {
            display: flex;
            margin-bottom: 16px;
            max-width: 80%;
        }

        /* Chat input text and placeholder styling */
        .stChatInput input::placeholder {
            color: #8c7b6b !important; /* 深米色 */
            opacity: 1 !important;
            font-size: 16px;
        }

        .stChatInput textarea::placeholder {
            color: #8c7b6b !important; /* 深米色 */
            opacity: 1 !important;
            font-size: 16px;
        }

        .stChatInput input {
            color: #4e4e4e !important; /* 深灰色文字 */
            font-size: 16px;
            caret-color: #39605a !important; /* 绿色光标 */
        }

        .stChatInput textarea {
            color: #4e4e4e !important; /* 深灰色文字 */
            font-size: 16px;
            caret-color: #39605a !important; /* 绿色光标 */
        }
        
        /* 聊天输入框样式 */
        .stChatInput > div {
            border-color: #8c7b6b !important; /* 深米色边框 */
            background-color: rgba(255, 255, 255, 0.9) !important; /* 半透明白色背景 */
            border-radius: 20px !important;
        }
        
        /* 输入框内部背景色 */
        .stChatInput input, .stChatInput textarea {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #4e4e4e !important;
        }
        
        /* 输入框聚焦状态 */
        .stChatInput div[data-testid="stChatInput"]:focus-within {
            border-color: #39605a !important; /* 绿色焦点 */
            box-shadow: 0 0 0 2px rgba(57, 96, 90, 0.3) !important;
        }
        
        /* User message container - align right */
        .user-container {
            margin-left: auto;
            justify-content: flex-end;
        }
        
        /* Assistant message container - align left */
        .assistant-container {
            margin-right: auto;
            justify-content: flex-start;
        }
        
        /* Message bubble styling */
        .message-bubble {
            padding: 12px 16px;
            border-radius: 16px;
            word-wrap: break-word;
        }
        
        /* User message styling */
        .user-bubble {
            background-color: rgba(255, 255, 255, 0.9); /* 半透明白色 */
            color: #4e4e4e; /* 深灰色文字 */
            border-radius: 16px 16px 0 16px;
            border: 2px solid #39605a !important; /* 绿色边框 */
        }
        
        /* Assistant message styling */
        .assistant-bubble {
            background-color: #39605a; /* 绿色背景 */
            color: white; /* 白色文字 */
            border-radius: 16px 16px 16px 0;
            border: 2px solid #4e4e4e !important; /* 深灰色边框 */
        }
                
        .stChatMessage:has([data-testid="stChatMessageAvatarCustom"]) {
            display: flex;
            flex-direction: row-reverse;
            align-self: end;
            background-color: rgba(255, 255, 255, 0.9);
            color: #4e4e4e;
            border-radius: 16px 16px 0 16px;
            border: 2px solid #39605a !important;
        }
        
        [data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
            text-align: right;
        }
                
        [class*="st-key-user"] {
            dispay: flex;
            flex-direction: row-reverse;
            p {
                font-size: 1.125rem;
                color: #4e4e4e;
                font-weight: medium;
            }
                
        }
                
        .stChatMessage {
            background-color: transparent;
        }

        [class*="st-key-assistant"] {
            background-color: #39605a; /* 绿色背景 */
            border-radius: 16px 16px 16px 0;
            padding-right: 16px;
            border: 2px solid #4e4e4e !important; /* 深灰色边框 */
                
            p {
                font-size: 1.125rem;
                color: white;
                font-weight: medium;
                padding-left: 4px;
            }
                
            img {
                display: flex;
                height: 52px;
                width: 52px;
            }
        }
        
        .st-key-chat_section{
            display: flex;
            flex-direction: column-reverse;
            justify-content: flex-end;
        }
        
        /* Remove red border outline from chat input when active */
        .stChatInput div[data-testid="stChatInput"] > div:focus-within {
            box-shadow: none !important;
            border-color: #39605a !important; /* 绿色 */
            border-width: 1px !important;
        }
        
        /* Change chat input focus state */
        .stChatInput div[data-testid="stChatInput"]:focus-within {
            border-color: #39605a !important; /* 绿色 */
            box-shadow: 0 0 0 1px rgba(57, 96, 90, 0.5) !important;
        }
        
        /* Remove default Streamlit outlines */
        *:focus {
            outline: none !important;
        }
        
        /* Target specifically the chat input elements */
        [data-testid="stChatInput"] input:focus {
            box-shadow: none !important;
            outline: none !important;
            border-color: #39605a !important; /* 绿色 */
        }
        
        [data-testid="stChatInput"] textarea:focus {
            box-shadow: none !important;
            outline: none !important;
            border-color: #39605a !important; /* 绿色 */
        }
        
        /* 主要按钮样式 */
        button[kind="primary"] {
            background-color: #39605a !important; /* 绿色 */
            color: white !important;
            border: 1px solid #af9b8a !important; /* 细浅灰色边框 */
            border-radius: 8px !important;
        }
        
        button[kind="primary"]:hover {
            background-color: #2d4d47 !important; /* 深绿色悬停 */
            border-color: #af9b8a !important;
        }
        
        /* 次要按钮样式 */
        button[kind="secondary"] {
            background-color: #4e4e4e !important; /* 深灰色 */
            color: white !important;
            border: 1px solid #af9b8a !important; /* 细浅灰色边框 */
            border-radius: 8px !important;
        }
        
        button[kind="secondary"]:hover {
            background-color: #3a3a3a !important; /* 更深灰色悬停 */
            border-color: #af9b8a !important;
        }
        
        /* Style the selected value in the dropdown */
        .stSelectbox div[data-baseweb="select"] > div {
            color: #4e4e4e !important;
            background-color: rgba(255, 255, 255, 0.9) !important;
        }
        
        /* 贴纸奖励样式 */
        .sticker-reward {
            background-color: rgba(227, 197, 170, 0.8); /* 与主背景相近的浅色，带透明度 */
            border: 2px solid #4e4e4e; /* 绿色边框 */
            border-radius: 15px;
            padding: 15px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(78, 78, 78, 0.1);
        }
        
        .sticker-reward img {
            width: 200px;
            border-radius: 10px;
        }
        
        .sticker-caption {
            font-size: 16px;
            margin-top: 10px;
            font-weight: bold;
            color: #4e4e4e;
        }
        
        /* 礼物盒子样式 */
        .gift-box {
            text-align: center;
            margin-top: 10px;
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            border: 3px solid #39605a;
        }
        
        .gift-box img {
            width: 120px;
            margin-top: 10px;
        }  
        
        /* 友谊分数区域 */
        .friendship-score {
            margin-bottom: 32px;
            padding: 24px;
            border-radius: 16px;
            background-color: transparent; /* 透明背景 */
            border: 2px solid transparent; /* 透明边框 */
            box-shadow: none; /* 移除阴影 */
        }

        
        .score-guide {
            position: fixed;
            bottom: 120px;
            left: calc(45% - 37%);
            width: 30%;
            background-color: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(78, 78, 78, 0.2);
            z-index: 101;
            border: 2px solid #39605a;
        }
        
        .close-btn {
            position: absolute;
            top: 5px;
            right: 5px;
            background: none;
            border: none;
            font-size: 16px;
            cursor: pointer;
            color: #4e4e4e;
        }
        
        /* 加载动画 */
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 10px;
            margin-top: 10px;
        }
        
        .loading-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #39605a; /* 绿色加载动画 */
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* 扩展面板样式 */
        .streamlit-expanderHeader {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #4e4e4e !important;
            border: 1px solid #39605a !important;
            border-radius: 10px !important;
        }
        
        .streamlit-expanderContent {
            background-color: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid #39605a !important;
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
        }
        
        /* 对话框样式 */
        .stDialog {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border: 2px solid #39605a !important;
            border-radius: 15px !important;
        }
        
        /* 侧边栏样式 */
        .stSidebar {
            background-color: #e3c5aa !important;
        }
        
        /* 响应式调整 */
        @media (max-width: 768px) {
            .sticker-reward img {
                width: 150px;
            }
            
            .friendship-score {
                padding: 16px;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    role = list(role_configs.keys())[0]
    role_config = role_configs[role]

    left_col, right_col = st.columns([0.63, 0.37], vertical_alignment="top", gap="large")
    
    with left_col:
        with open("rock.png", "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0; padding: 0;">
                <div style="display: flex;">
                    <img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 200px;">
                </div>
                <div style="flex: 1;">
                    <h1 style="margin-top: 0; font-size: 3rem; padding: 0;">{texts['title']}</h1>
                    <h1 style="margin-top: 0; font-size: 3rem; padding: 0;">{texts['subtitle']}</h1>
                    <h3 style="margin-top: 0.5rem; font-weight: bold; padding: 0; font-size: 1.25rem;">{texts['prompt']}</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Chat input (full width under title)
        user_input = st.chat_input(placeholder=texts['chat_placeholder'])
        print(f"User input: {user_input}")
        
        # Chat Section
        chatSection = st.container(key="chat_section", border=False)
        with chatSection:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            for message in st.session_state.chat_history:
                with chat_message(message["role"]):
                    st.markdown(message["content"])
        

        if user_input and user_input != st.session_state.last_question:
            try:
                # Set processing state first
                st.session_state.processing = True
                st.session_state.has_interacted = True
                st.session_state.show_score_guide = False
                # Store the input for this session
                current_input = user_input
                
                # Add to chat history immediately
                st.session_state.chat_history.append({"role": "user", "content": current_input})
                st.session_state.last_question = current_input
                
                # Display user message
                with chatSection:
                    with chat_message("user"):
                        st.markdown(current_input)
                
                with chatSection:
                    loading_placeholder = st.empty()
                    with st.spinner(""):
                        loading_placeholder.markdown(f"""
                            <div class="loading-container">
                                <div class="loading-spinner"></div>
                                <div>{texts['loading_thought']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Process response
                try:
                    # 使用优化的 RAG 检索器（带缓存）
                    rag = get_rag_instance(
                        persist_directory=get_vectordb(role),
                        dashscope_api_key=dashscope_key
                    )

                    if st.session_state.language == "Portuguese":
                        k_value = 2  # Fewer documents for OpenAI
                    else:
                        k_value = 4
                    
                    # 智能检索：动态 k 值、相关性过滤
                    most_relevant_texts = rag.retrieve(
                        query=current_input,
                        lambda_mult=0.3,  # 优先相关性（从0.7降到0.3）
                        relevance_threshold=None  # 暂不启用过滤
                    )
                    if st.session_state.language == "Portuguese":
                        print(f"[Processing] Truncating documents for Portuguese to avoid token limits")
                        most_relevant_texts = truncate_documents_for_portuguese(most_relevant_texts, max_chars=1200)
                    chain, role_config = get_conversational_chain(role, st.session_state.language)
                    # 优化：使用 invoke() 替代弃用的 run()
                    raw_answer = chain.invoke({"input_documents": most_relevant_texts, "question": current_input})
                    # 处理 invoke() 返回的字典格式
                    answer_text = raw_answer.get("output_text", raw_answer) if isinstance(raw_answer, dict) else raw_answer
                    answer = re.sub(r'^\s*Answer:\s*', '', answer_text).strip()
                    st.session_state.last_answer = answer

                    # Save results to session state
                    st.session_state.most_relevant_texts = most_relevant_texts
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    update_intimacy_score(current_input)
                    gift_triggered = check_gift()

                    # Generate and play audio
                    speak_text(answer, loading_placeholder)
                    
                    # Display assistant message
                    with chatSection:
                        with chat_message("assistant"):
                            st.markdown(answer)
                            
                    st.session_state.audio_played = True
                    st.session_state.processing = False
                    
                except Exception as e:
                    # Handle processing errors
                    print(f"Error processing response: {str(e)}")
                    if loading_placeholder:
                        loading_placeholder.empty()
                        
                    error_msg = texts['error_message']
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    
                    with chatSection:
                        with chat_message("assistant"):
                            st.markdown(error_msg)
                            st.error(f"Error details: {str(e)}")
            
            except Exception as outer_e:
                # Handle any unexpected errors
                print(f"Outer exception in user input handling: {str(outer_e)}")
                st.error(f"An unexpected error occurred: {str(outer_e)}")


        # Gift section
        @st.dialog("🎁 Your Gift", width=680)
        def gift_dialog():
            with open("gift.png", "rb") as f:
                gift_img_base64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <div class="petrel-response gift-box">
                    <p>{texts['gift_message']}</p>
                    <img src="data:image/png;base64,{gift_img_base64}">
                    <div class="sticker-caption">{texts['medal_caption']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        if st.session_state.gift_given and not st.session_state.gift_shown: 
            gift_dialog()
            st.session_state.gift_shown = True
            
        

    with right_col:
        # Language switcher
        st.markdown("**Language / Idioma:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🇬🇧 English", use_container_width=True, 
                        type="primary" if st.session_state.language == "English" else "secondary"):
                st.session_state.language = "English"
                st.rerun()
        with col2:
            if st.button("🇵🇹 Português", use_container_width=True,
                        type="primary" if st.session_state.language == "Portuguese" else "secondary"):
                st.session_state.language = "Portuguese"
                st.rerun()
        
        # Tips and Clear buttons
        input_section_col1, input_section_col2 = st.columns([0.35, 0.65], gap="small")
        with input_section_col1:
            # Show guide if toggled
            @st.dialog(texts['score_guide_title'], width="large") 
            def score_guide():
                st.markdown(texts['tips_content'], unsafe_allow_html=True)
                
            if st.button(texts['tips_button'], icon=":material/lightbulb:", 
                        help=texts['tips_help'], 
                        use_container_width=True, type="primary"):
                score_guide()
                
        with input_section_col2:
            if st.button(texts['clear_button'], icon=":material/chat_add_on:", 
                        help=texts['clear_help'],  
                        use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.show_score_guide = False
                st.session_state.audio_played = True
                st.session_state.gift_given = False
                st.session_state.intimacy_score = 0
                st.session_state.awarded_stickers = []
                st.session_state.last_question = ""
                st.session_state.has_interacted = False
                st.session_state.processing = False
                st.session_state.most_relevant_texts = []
                st.session_state.last_answer = ""
                st.session_state.last_sticker = None
                st.session_state.last_analysis = {}
                st.session_state.newly_awarded_sticker = False
                st.session_state.gift_shown = False
                if "session_id" in st.session_state:
                    del st.session_state["session_id"]
                if "logged_interactions" in st.session_state:
                    del st.session_state["logged_interactions"]
                st.rerun()
        
        # Friendship score section
        current_score = min(6, int(round(st.session_state.intimacy_score)))
        
        st.markdown(f"""
        <div class="friendship-score">
            <div style="font-size:18px; font-style: italic; font-weight:bold; color:#31333e; text-align: left;">
                {texts['friendship_score']}
            </div>
            <div style="font-size:16px; color:#31333e; text-align: left;">{texts['score_description']}</div>
            <div style="font-size:24px; margin:5px 0; text-align: left;">
                <span style="color:#ff6b6b;">{'❤️' * current_score}</span>
                <span style="color:#ddd;">{'🤍' * (6 - current_score)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sticker Shown
        if st.session_state.last_question and user_input:
            normalized_input = st.session_state.last_question.strip().lower()
            
            st.session_state.newly_awarded_sticker = False
            
            # Check if this question matches any sticker criteria
            for q, reward in sticker_rewards.items():
                exact = q.lower() == normalized_input

                is_semantic_match = semantic_match(normalized_input, q, reward)

                keywords = reward.get("semantic_keywords", [])
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in normalized_input)
                keyword_match = keyword_matches >= 2
                print(f"Checking question: '{q}' | Exact match: {exact} | Semantic match: {is_semantic_match} | Keyword matches: {keyword_matches} (required: 2)")
                if exact or is_semantic_match or keyword_match:
                    # Add this sticker to the awarded list if not already present
                    sticker_key = reward["image"]
                    if sticker_key not in [s["key"] for s in st.session_state.awarded_stickers]:
                        # Use language-specific caption if available
                        caption = reward["caption"][st.session_state.language] if isinstance(reward["caption"], dict) else reward["caption"]
                        st.session_state.awarded_stickers.append({
                            "key": sticker_key,
                            "image": reward["image"],
                            "caption": caption
                        })
                        st.toast(texts['sticker_toast'], icon="⭐")
                        st.session_state.newly_awarded_sticker = True
                    break
        # Display the most recent sticker if any exist
        if st.session_state.awarded_stickers:
            # Get the most recent sticker (last in the list)
            most_recent = st.session_state.awarded_stickers[-1]

            current_caption = most_recent["caption"]
            for q, reward in sticker_rewards.items():
                if reward["image"] == most_recent["image"]:
                    if isinstance(reward["caption"], dict) and st.session_state.language in reward["caption"]:
                        current_caption = reward["caption"][st.session_state.language]
                    break

            st.markdown(
                f"""
                <div class="sticker-reward">
                    <img src="data:image/png;base64,{base64.b64encode(open(most_recent["image"], "rb").read()).decode()}">
                    <div class="sticker-caption">{current_caption}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Add a small indicator showing how many stickers have been collected
            total_possible = len(sticker_rewards)
            total_collected = len(st.session_state.awarded_stickers)
            
            st.markdown(
                f"""
                <div style="text-align: center; font-size: 14px; margin-top: -10px; color: #555; margin-bottom: 20px;">
                    {texts['stickers_collected'].format(current=total_collected, total=total_possible)}
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Fact Check Section
        st.markdown(f"""
            <div style="font-size:18px; font-style: italic; font-weight:bold; color:#31333e; text-align: left;">
                {texts['doubtful']}
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander(texts['fact_check'], expanded=False):
            if "most_relevant_texts" in st.session_state and "last_question" in st.session_state and "last_answer" in st.session_state:
                # 生成智能摘要（替代原始文档内容）
                if len(st.session_state.most_relevant_texts) > 0:
                    try:
                        fact_check_summary = generate_fact_check_content(
                            question=st.session_state.last_question,
                            retrieved_docs=st.session_state.most_relevant_texts,
                            ai_answer=st.session_state.last_answer,
                            language=st.session_state.language
                        )
                        
                        # 使用容器样式包裹 Markdown 内容
                        st.markdown("""
                            <style>
                            .fact-check-box {
                                background: #f0f8ff;
                                padding: 20px;
                                border-radius: 10px;
                                margin: 10px 0;
                                border-left: 4px solid #4a90e2;
                                color: #2c3e50;
                                line-height: 1.6;
                            }
                            .fact-check-box p {
                                margin-bottom: 10px;
                            }
                            .fact-check-box strong {
                                color: #1e3a8a;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        # 直接使用 st.markdown 渲染，应用样式类
                        #st.markdown(f'<div class="fact-check-box">', unsafe_allow_html=True)
                        st.markdown(fact_check_summary)
                        st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        # 降级：显示原始内容
                        print(f"[Fact-Check] 摘要生成失败: {str(e)}")
                        st.write(st.session_state.most_relevant_texts[0].page_content[:300] + "...")
            else:
                st.info(texts['fact_check_info'])
    cleanup_audio_files()

    # Log the interaction to Supabase
    if st.session_state.last_question:
        # Check if this specific interaction has already been logged
        if "logged_interactions" not in st.session_state:
            st.session_state.logged_interactions = set()
        
        combined = f"{st.session_state.last_question}|{st.session_state.last_answer}"

        interaction_key = hashlib.md5(combined.encode()).hexdigest()
        if interaction_key not in st.session_state.logged_interactions:
            log_interaction(
                user_input=st.session_state.last_question,
                ai_response=st.session_state.last_answer,
                intimacy_score=st.session_state.intimacy_score,
                is_sticker_awarded=st.session_state.newly_awarded_sticker,
                gift_given=st.session_state.gift_given
            )
            st.session_state.logged_interactions.add(interaction_key)

if __name__ == "__main__":
    main()
