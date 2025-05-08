import streamlit as st
import tempfile
import wave
import numpy as np
from transformers import pipeline as hf_pipeline
from audio_recorder_streamlit import audio_recorder
import whisper
from googletrans import Translator

# App setup
st.set_page_config(page_title="Hindi Audio Processor", layout="centered")
st.title("🎙️ Hindi/Hinglish Audio Processor")

# Initialize models (cached)
@st.cache_resource
def load_models():
    try:
        # Load Whisper model (using tiny for CPU efficiency)
        model = whisper.load_model("tiny")
        
        # Initialize translator
        translator = Translator()
        
        # Load sentiment analysis model explicitly
        sentiment = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        return model, translator, sentiment
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Load models
whisper_model, translator, sentiment_analyzer = load_models()

def create_proper_wav(audio_bytes, sample_rate=16000):
    """Convert raw audio bytes to proper WAV format without FFmpeg"""
    try:
        # Convert to numpy array (16-bit PCM)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(audio_array.tobytes())
            return tmp.name
    except Exception as e:
        st.error(f"Audio conversion failed: {str(e)}")
        return None

def process_audio(audio_path):
    try:
        # Transcribe with Whisper
        result = whisper_model.transcribe(audio_path)
        hindi_text = result["text"]
        
        # Translate to English
        english_text = translator.translate(hindi_text, src='hi', dest='en').text
        
        # Generate summary
        sentences = [s.strip() for s in english_text.split('.') if s.strip()]
        summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else english_text
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer(english_text[:512])[0]
        
        return {
            "Hindi Transcript": hindi_text,
            "English Translation": english_text,
            "Summary": summary,
            "Sentiment": f"{sentiment_result['label']} ({sentiment_result['score']:.0%})"
        }
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# Main app
def main():
    # Audio input
    st.sidebar.header("Input Options")
    input_method = st.sidebar.radio("Select:", ("Record", "Upload WAV"))
    
    audio_data = None
    
    if input_method == "Record":
        audio_bytes = audio_recorder(
            pause_threshold=5.0,
            sample_rate=16000,
            text="Click to record"
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            audio_data = audio_bytes
    else:
        uploaded_file = st.file_uploader("Upload 16-bit mono WAV", type=["wav"])
        if uploaded_file:
            audio_data = uploaded_file.read()
            st.audio(audio_data, format="audio/wav")

    if audio_data and st.button("🚀 Process", type="primary"):
        with st.spinner("Processing..."):
            # Create proper WAV file
            wav_path = create_proper_wav(audio_data)
            
            if wav_path:
                results = process_audio(wav_path)
                
                # Clean up
                try:
                    os.unlink(wav_path)
                except:
                    pass
                
                if results:
                    st.subheader("Results")
                    st.json(results)
                    
                    st.download_button(
                        "📥 Download",
                        str(results),
                        file_name="results.json"
                    )

if __name__ == "__main__":
    import os
    main()
