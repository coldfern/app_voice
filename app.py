import streamlit as st
import tempfile
import os
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
        # Load Whisper model (using small for balance of speed/accuracy)
        whisper_model = whisper.load_model("small")
        
        # Initialize translator
        translator = Translator()
        
        # Explicitly load sentiment analysis model
        sentiment_analyzer = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        return whisper_model, translator, sentiment_analyzer
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

# Load models when app starts
whisper_model, translator, sentiment_analyzer = load_models()

def create_valid_wav(audio_bytes, sample_rate=16000):
    """Convert raw audio bytes to valid WAV format"""
    try:
        # Convert to numpy array (assuming 16-bit PCM)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_array.tobytes())
            return tmp.name
    except Exception as e:
        st.error(f"Audio conversion failed: {str(e)}")
        return None

# Audio processing function
def process_audio(audio_path):
    try:
        # Transcribe with Whisper
        result = whisper_model.transcribe(audio_path)
        hindi_text = result["text"]
        
        # Translate to English
        english_text = translator.translate(hindi_text, src='hi', dest='en').text
        
        # Generate summary (first 2 meaningful sentences)
        sentences = [s.strip() for s in english_text.split('.') if len(s.strip()) > 10]
        summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else english_text
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer(english_text[:512])[0]
        
        return {
            "Hindi Transcript": hindi_text,
            "English Translation": english_text,
            "Summary": summary,
            "Sentiment": f"{sentiment_result['label']} ({sentiment_result['score']:.0%} confidence)"
        }
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# Main app interface
def main():
    # Audio input options
    st.sidebar.header("Input Options")
    input_method = st.sidebar.radio("Select input method:", ("Record Audio", "Upload Audio File"))

    audio_data = None
    
    if input_method == "Record Audio":
        st.write("Speak in Hindi/Hinglish (minimum 5 seconds):")
        audio_bytes = audio_recorder(
            pause_threshold=5.0,
            sample_rate=16000,
            text="Click to record"
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            audio_data = audio_bytes
    else:
        uploaded_file = st.file_uploader("Upload WAV file (16-bit mono, 16kHz)", type=["wav"])
        if uploaded_file:
            audio_data = uploaded_file.read()
            st.audio(audio_data, format="audio/wav")

    # Process audio when available
    if audio_data:
        if st.button("🚀 Process Audio", type="primary"):
            with st.spinner("Processing your audio..."):
                # Create valid WAV file
                audio_path = create_valid_wav(audio_data)
                
                if audio_path:
                    # Process the audio file
                    results = process_audio(audio_path)
                    
                    # Clean up temp file
                    try:
                        os.unlink(audio_path)
                    except:
                        pass
                    
                    # Display results
                    if results:
                        st.subheader("Results")
                        st.json(results)
                        
                        # Download button
                        st.download_button(
                            "💾 Download Results",
                            str(results),
                            file_name="audio_analysis_results.json",
                            mime="application/json"
                        )

if __name__ == "__main__":
    import wave
    main()
