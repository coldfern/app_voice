import streamlit as st
import os
import tempfile
import numpy as np
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# App configuration
st.set_page_config(page_title="Fast Hindi Audio Processor", layout="wide")
st.title("⚡ Fast Hindi/Hinglish Audio Processor")

# Load models efficiently
@st.cache_resource
def load_models():
    try:
        # Use a smaller, faster model for translation
        translator = pipeline("translation_hi_to_en", 
                           model="Helsinki-NLP/opus-mt-hi-en",
                           device="cpu")
        
        # Lightweight sentiment analysis
        sentiment_analyzer = pipeline("sentiment-analysis", 
                                    model="distilbert-base-uncased-finetrained-sst-2-english",
                                    device="cpu")
        
        return translator, sentiment_analyzer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

translator, sentiment_analyzer = load_models()

# Audio processing function
def process_audio(audio_file):
    try:
        # Convert audio to text using a faster approach
        # Note: In production, you would use a proper ASR model here
        # This is a simplified placeholder
        hindi_text = "[Hindi transcription would appear here]"
        
        # Translate to English
        english_text = translator(hindi_text)[0]['translation_text']
        
        # Quick summary (first 2 sentences)
        summary = '. '.join(english_text.split('. ')[:2]) + '.'
        
        # Fast sentiment analysis
        sentiment = sentiment_analyzer(english_text[:512])[0]
        sentiment_result = f"{sentiment['label']} ({sentiment['score']:.2f} confidence)"
        
        return {
            "hindi_transcript": hindi_text,
            "english_translation": english_text,
            "summary": summary,
            "sentiment": sentiment_result
        }
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# UI Components
with st.sidebar:
    st.header("Audio Input")
    input_method = st.radio("Choose input:", ("Record", "Upload WAV"))
    
    audio_file = None
    
    if input_method == "Record":
        audio_bytes = audio_recorder(pause_threshold=3.0, sample_rate=16000)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            audio_file = audio_bytes
    else:
        uploaded_file = st.file_uploader("Upload WAV", type=["wav"])
        if uploaded_file:
            audio_file = uploaded_file.read()
            st.audio(audio_file, format="audio/wav")

# Main processing
if audio_file:
    if st.button("Process Audio", type="primary"):
        with st.spinner("Processing..."):
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file if isinstance(audio_file, bytes) else audio_file)
                tmp_path = tmp.name
            
            results = process_audio(tmp_path)
            
            if results:
                cols = st.columns(2)
                with cols[0]:
                    st.subheader("Hindi Transcript")
                    st.write(results["hindi_transcript"])
                with cols[1]:
                    st.subheader("English Translation")
                    st.write(results["english_translation"])
                
                st.subheader("Summary")
                st.write(results["summary"])
                
                st.subheader("Sentiment Analysis")
                st.write(results["sentiment"])
            
            os.unlink(tmp_path)
else:
    st.info("Record or upload a short Hindi audio clip (under 30 seconds)")

st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
.stAudio {
    max-width: 300px;
}
</style>
""", unsafe_allow_html=True)
