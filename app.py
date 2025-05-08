import streamlit as st
import os
import tempfile
import wave
import numpy as np
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import whisper
import warnings
from io import BytesIO

# Suppress warnings
warnings.filterwarnings("ignore")

# App title and config
st.set_page_config(page_title="Hindi/Hinglish Audio Processor", layout="wide")
st.title("🇮🇳 Hindi/Hinglish Audio Processor")

# Initialize models (cache them for performance)
@st.cache_resource
def load_models():
    try:
        # Whisper model for transcription (tiny is fastest)
        whisper_model = whisper.load_model("tiny")
        
        # Translation pipeline
        translator = pipeline("translation", 
                            model="Helsinki-NLP/opus-mt-hi-en",
                            device="cpu")
        
        # Sentiment analysis pipeline
        sentiment_analyzer = pipeline("sentiment-analysis", 
                                    model="distilbert-base-uncased-finetuned-sst-2-english",
                                    device="cpu")
        
        return whisper_model, translator, sentiment_analyzer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

whisper_model, translator, sentiment_analyzer = load_models()

def create_compatible_wav(audio_data, sample_rate=16000):
    """Create a compatible WAV file from raw audio data"""
    try:
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Create in-memory WAV file
        with BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            wav_data = wav_buffer.getvalue()
        
        return wav_data
    except Exception as e:
        st.error(f"Audio conversion failed: {str(e)}")
        return None

# Function to process audio
def process_audio(audio_file_path):
    try:
        # Transcribe with Whisper
        result = whisper_model.transcribe(audio_file_path)
        hindi_text = result["text"]
        
        # Translate to English
        english_text = translator(hindi_text)[0]['translation_text']
        
        # Generate summary
        sentences = english_text.split('. ')
        summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else english_text
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer(english_text[:512])[0]
        sentiment = f"{sentiment_result['label']} (confidence: {sentiment_result['score']:.2f})"
        
        return {
            "hindi_transcript": hindi_text,
            "english_translation": english_text,
            "summary": summary,
            "sentiment": sentiment
        }
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# Audio input section
with st.sidebar:
    st.header("Audio Input")
    input_method = st.radio("Choose input method:", ("Record Audio", "Upload WAV File"))
    
    audio_file = None
    
    if input_method == "Record Audio":
        st.write("Please record in Hindi/Hinglish (minimum 5 seconds):")
        audio_bytes = audio_recorder(pause_threshold=5.0, sample_rate=16000)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            # Create compatible WAV file
            wav_data = create_compatible_wav(audio_bytes)
            if wav_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(wav_data)
                    audio_file = tmp.name
    else:
        uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
        if uploaded_file:
            # Verify it's a proper WAV file
            try:
                with wave.open(uploaded_file, 'rb') as wav:
                    if wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 16000:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(uploaded_file.read())
                            audio_file = tmp.name
                        st.audio(audio_file, format="audio/wav")
                    else:
                        st.error("Please upload a 16-bit mono WAV file at 16kHz sample rate")
            except:
                st.error("Invalid WAV file. Please upload a proper 16-bit mono WAV file at 16kHz")

# Main processing
if audio_file:
    if st.button("Process Audio"):
        with st.spinner("Processing audio..."):
            results = process_audio(audio_file)
            
            if results:
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Hindi Transcript", "English Translation", 
                    "Summary", "Sentiment Analysis"
                ])
                
                with tab1:
                    st.subheader("Hindi Transcript")
                    st.write(results["hindi_transcript"])
                    st.download_button(
                        "Download Transcript",
                        results["hindi_transcript"],
                        file_name="hindi_transcript.txt"
                    )
                
                with tab2:
                    st.subheader("English Translation")
                    st.write(results["english_translation"])
                    st.download_button(
                        "Download Translation",
                        results["english_translation"],
                        file_name="english_translation.txt"
                    )
                
                with tab3:
                    st.subheader("Summary")
                    st.write(results["summary"])
                    st.download_button(
                        "Download Summary",
                        results["summary"],
                        file_name="summary.txt"
                    )
                
                with tab4:
                    st.subheader("Sentiment Analysis")
                    st.write(results["sentiment"])
            
        if os.path.exists(audio_file):
            os.unlink(audio_file)
else:
    st.info("Please record or upload a WAV audio file")

st.markdown("""
### Instructions:
1. Record or upload a WAV audio file (16-bit mono, 16kHz sample rate)
2. Audio should be in Hindi or Hinglish
3. Click "Process Audio"
4. View results in different tabs
5. Download any results as text files

Note: First run will take 2-3 minutes to download models.
""")
