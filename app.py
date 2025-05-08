import streamlit as st
import tempfile
import numpy as np
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder

# Configure app for speed
st.set_page_config(page_title="⚡ Hindi Audio Processor", layout="centered")
st.title("⚡ Hindi/Hinglish Audio Processor")

# Load FAST models (cached)
@st.cache_resource
def load_models():
    # Tiny translation model (faster than Helsinki-NLP)
    translator = pipeline("translation", 
                        model="facebook/m2m100_418M",
                        device="cpu")
    
    # Tiny sentiment model
    sentiment = pipeline("sentiment-analysis", 
                       model="finiteautomata/bertweet-base-sentiment-analysis",
                       device="cpu")
    
    return translator, sentiment

translator, sentiment_analyzer = load_models()

# Mock ASR function (replace with your fast ASR API)
def transcribe_audio(audio_path):
    # In production: Connect to fast ASR service like:
    # - Whisper-API
    # - Google Speech-to-Text
    # - Azure Speech Service
    return "यह हिंदी में एक उदाहरण पाठ है"

def process_audio(audio_bytes):
    try:
        # Step 1: Transcribe (mock)
        hindi_text = transcribe_audio(audio_bytes)
        
        # Step 2: Translate
        english_text = translator(hindi_text[:512])[0]['translation_text']
        
        # Step 3: Summarize (first sentence)
        summary = english_text.split('.')[0] + '.'
        
        # Step 4: Sentiment
        sentiment = sentiment_analyzer(english_text[:256])[0]
        
        return {
            "hindi": hindi_text,
            "english": english_text,
            "summary": summary,
            "sentiment": f"{sentiment['label']} ({sentiment['score']:.0%})"
        }
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# UI - Minimal design for speed
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        audio_bytes = audio_recorder(text="🎤 Record (5-30s)", 
                                  pause_threshold=5.0,
                                  sample_rate=16000)
        
    with col2:
        uploaded_file = st.file_uploader("Or upload WAV", type=["wav"])

# Process on button click
if audio_bytes or uploaded_file:
    audio_data = audio_bytes if audio_bytes else uploaded_file.read()
    
    if st.button("🚀 Process", type="primary"):
        with st.spinner("Processing..."):
            result = process_audio(audio_data)
            
            if result:
                st.subheader("Results")
                st.json({
                    "Hindi Text": result["hindi"],
                    "English Translation": result["english"],
                    "Summary": result["summary"],
                    "Sentiment": result["sentiment"]
                })

                st.download_button(
                    "📥 Download Results",
                    str(result),
                    file_name="audio_results.json"
                )
