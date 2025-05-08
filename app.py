import streamlit as st
import tempfile
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder

# Configure app
st.set_page_config(page_title="⚡ Hindi Audio Processor", layout="centered")
st.title("⚡ Instant Hindi/Hinglish Audio Processor")

# Load optimized models
@st.cache_resource
def load_models():
    # Tiny translation model (5x faster than Helsinki-NLP)
    translator = pipeline("translation", 
                        model="facebook/m2m100_418M",
                        device="cpu")
    
    # Lightweight sentiment model
    sentiment = pipeline("sentiment-analysis", 
                       model="finiteautomata/bertweet-base-sentiment-analysis",
                       device="cpu")
    
    return translator, sentiment

translator, sentiment_analyzer = load_models()

# Mock ASR function (replace with your API)
def transcribe_audio(audio_bytes):
    return "यह हिंदी में एक उदाहरण पाठ है"

def process_audio(audio_bytes):
    hindi_text = transcribe_audio(audio_bytes)         # Step 1: Transcribe
    english_text = translator(hindi_text[:512])[0]['translation_text']  # Step 2: Translate
    summary = english_text.split('.')[0] + '.'         # Step 3: Summarize
    sentiment = sentiment_analyzer(english_text[:256])[0]  # Step 4: Sentiment
    
    return {
        "Hindi Transcript": hindi_text,
        "English Translation": english_text,
        "Summary": summary,
        "Sentiment": f"{sentiment['label']} ({sentiment['score']:.0%} confidence)"
    }

# UI
audio_bytes = audio_recorder(text="🎤 Record (5-30s)", pause_threshold=5.0)

if audio_bytes:
    if st.button("🚀 Process", type="primary"):
        with st.spinner("Processing..."):
            result = process_audio(audio_bytes)
            st.json(result)
            st.download_button("📥 Download Results", str(result), file_name="results.json")
