import streamlit as st
import tempfile
import numpy as np
import wave
from transformers import pipeline
from googletrans import Translator
from audio_recorder_streamlit import audio_recorder
import os

# App setup
st.set_page_config(page_title="Hindi Audio Processor", layout="centered")
st.title("🎙️ Hindi/Hinglish Audio Processor")

# Initialize models (cached)
@st.cache_resource
def load_models():
    try:
        # Load automatic speech recognition (ASR) model
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="ai4bharat/indicwav2vec-hindi"
        )
        
        # Initialize translator
        translator = Translator()
        
        # Load sentiment analysis model
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        return asr_pipeline, translator, sentiment_analyzer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Load models
asr_model, translator, sentiment_analyzer = load_models()

def process_audio(audio_path):
    try:
        # Transcribe with IndicWav2Vec
        hindi_text = asr_model(audio_path)["text"]
        
        # Translate to English
        english_text = translator.translate(hindi_text, src='hi', dest='en').text
        
        # Generate summary (first 2 sentences)
        sentences = [s.strip() for s in english_text.split('.') if s.strip()]
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

# Main app
def main():
    # Audio input
    audio_bytes = audio_recorder(
        pause_threshold=5.0,
        sample_rate=16000,
        text="Click to record Hindi/Hinglish"
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button("🚀 Process Audio", type="primary"):
            with st.spinner("Processing..."):
                # Save to temp WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    # Convert bytes to numpy array
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Write WAV file
                    with wave.open(tmp.name, 'wb') as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)  # 16-bit
                        wav.setframerate(16000)
                        wav.writeframes(audio_array.tobytes())
                    
                    # Process audio
                    results = process_audio(tmp.name)
                
                # Clean up
                try:
                    os.unlink(tmp.name)
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
