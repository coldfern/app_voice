import streamlit as st
import tempfile
import wave
import numpy as np
from transformers import pipeline as transformers_pipeline
import whisper
from googletrans import Translator
from audio_recorder_streamlit import audio_recorder
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# App setup
st.set_page_config(page_title="Hindi Audio Processor", layout="centered")
st.title("🎙️ Hindi/Hinglish Audio Processor")

# Initialize models (cached)
@st.cache_resource
def load_models():
    try:
        # Load Whisper model with explicit FP32
        model = whisper.load_model("base")
        
        # Initialize translator
        translator = Translator()
        
        # Load sentiment analysis model with explicit specification
        sentiment = transformers_pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt"
        )
        
        return model, translator, sentiment
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Load models
whisper_model, translator, sentiment_analyzer = load_models()

def create_valid_wav(audio_bytes, sample_rate=16000):
    """Convert raw audio to proper WAV format"""
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_array.tobytes())
            return tmp.name
    except Exception as e:
        st.error(f"Audio conversion failed: {str(e)}")
        return None

def process_audio(audio_path):
    try:
        # Transcribe with explicit FP32
        result = whisper_model.transcribe(audio_path, fp16=False)
        hindi_text = result["text"]
        
        # Translate
        english_text = translator.translate(hindi_text, src='hi', dest='en').text
        
        # Generate summary
        sentences = [s.strip() for s in english_text.split('.') if len(s.strip()) > 10]
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
    audio_bytes = audio_recorder(
        pause_threshold=5.0,
        sample_rate=16000,
        text="Click to record Hindi/Hinglish",
        recording_color="#e8b62c",
        neutral_color="#6aa36f"
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button("🚀 Process Audio", type="primary"):
            with st.spinner("Processing..."):
                # Convert to WAV
                wav_path = create_valid_wav(audio_bytes)
                
                if wav_path:
                    results = process_audio(wav_path)
                    
                    # Clean up
                    try:
                        os.unlink(wav_path)
                    except:
                        pass
                    
                    # Display results
                    if results:
                        st.subheader("Results")
                        st.json(results)
                        
                        st.download_button(
                            "📥 Download Results",
                            str(results),
                            file_name="hindi_audio_results.json"
                        )

if __name__ == "__main__":
    main()
