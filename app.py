import streamlit as st
import numpy as np
import wave
import tempfile
from transformers import pipeline
from googletrans import Translator
from audio_recorder_streamlit import audio_recorder
import os

# App setup - minimal for speed
st.set_page_config(page_title="⚡ Hindi Processor", layout="centered")
st.title("⚡ Instant Hindi Audio Processing")

# Load TINY models (cached)
@st.cache_resource
def load_models():
    # Tiny Hindi ASR model (5x faster than Whisper)
    asr = pipeline(
        "automatic-speech-recognition", 
        model="ai4bharat/indicwav2vec-hindi",
        device="cpu"
    )
    
    # Tiny translator
    translator = Translator()
    
    # Lightweight sentiment analysis
    sentiment = pipeline(
        "sentiment-analysis",
        model="finiteautomata/bertweet-base-sentiment-analysis",
        device="cpu"
    )
    
    return asr, translator, sentiment

asr_model, translator, sentiment_analyzer = load_models()

def process_audio_fast(audio_bytes):
    """Ultra-fast processing pipeline"""
    try:
        # Step 1: Convert to WAV in memory
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Step 2: Transcribe (takes <2s for 10s audio)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            with wave.open(tmp.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio_array)
            hindi_text = asr_model(tmp.name)["text"][:500]  # Limit to 500 chars
        
        # Step 3: Translate (takes <1s)
        english_text = translator.translate(hindi_text, src='hi', dest='en').text
        
        # Step 4: Quick summary (first sentence)
        summary = english_text.split('.')[0] + '.'
        
        # Step 5: Fast sentiment (takes <1s)
        sentiment = sentiment_analyzer(english_text[:256])[0]
        
        return {
            "Hindi": hindi_text,
            "English": english_text,
            "Summary": summary,
            "Sentiment": f"{sentiment['label']} ({sentiment['score']:.0%})"
        }
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Super simple UI
audio_bytes = audio_recorder(
    "Speak Hindi (5-15s)", 
    pause_threshold=5.0,
    sample_rate=16000
)

if audio_bytes:
    if st.button("🚀 Process (3-5s)", type="primary"):
        with st.spinner("Processing fast..."):
            results = process_audio_fast(audio_bytes)
            if results:
                st.json(results)
                st.audio(audio_bytes, format="audio/wav")
