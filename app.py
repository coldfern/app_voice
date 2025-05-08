import streamlit as st
import tempfile
import wave
import numpy as np
from transformers import pipeline
import whisper
from googletrans import Translator
from audio_recorder_streamlit import audio_recorder
import os

# App setup
st.set_page_config(page_title="Auto Hindi Transcriber", layout="centered")
st.title("🎙️ Auto Hindi/Hinglish Transcriber")

# Initialize models (cached)
@st.cache_resource
def load_models():
    try:
        # Load Whisper model (small for balance of speed/accuracy)
        model = whisper.load_model("small")
        
        # Initialize translator
        translator = Translator()
        
        # Load sentiment analysis
        sentiment = pipeline("sentiment-analysis")
        
        return model, translator, sentiment
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Load models
whisper_model, translator, sentiment_analyzer = load_models()

def audio_to_wav(audio_bytes, sample_rate=16000):
    """Convert raw audio bytes to proper WAV format"""
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
    # Record audio
    st.write("Speak in Hindi/Hinglish (minimum 5 seconds):")
    audio_bytes = audio_recorder(
        pause_threshold=5.0,
        sample_rate=16000,
        text="Click to record",
        recording_color="#e8b62c",
        neutral_color="#6aa36f"
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button("🚀 Transcribe & Analyze", type="primary"):
            with st.spinner("Processing..."):
                # Auto-convert to WAV
                wav_path = audio_to_wav(audio_bytes)
                
                if wav_path:
                    # Process the audio
                    results = process_audio(wav_path)
                    
                    # Clean up temp file
                    try:
                        os.unlink(wav_path)
                    except:
                        pass
                    
                    # Display results
                    if results:
                        st.subheader("Hindi Transcript")
                        st.write(results["Hindi Transcript"])
                        
                        st.subheader("English Translation")
                        st.write(results["English Translation"])
                        
                        st.subheader("Summary")
                        st.write(results["Summary"])
                        
                        st.subheader("Sentiment Analysis")
                        st.write(results["Sentiment"])
                        
                        # Download button
                        st.download_button(
                            "💾 Download Results",
                            str(results),
                            file_name="transcription_results.json"
                        )

if __name__ == "__main__":
    main()
