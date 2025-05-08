import streamlit as st
import os
import tempfile
import wave
import numpy as np
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import whisper
import warnings

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

def convert_to_compatible_wav(input_path, output_path):
    """Convert any WAV file to 16kHz mono 16-bit PCM format"""
    try:
        with wave.open(input_path, 'rb') as infile:
            params = infile.getparams()
            frames = infile.readframes(params.nframes)
            
            # Convert to numpy array
            audio_array = np.frombuffer(frames, dtype=np.int16)
            
            # Handle stereo files by averaging channels
            if params.nchannels == 2:
                audio_array = audio_array.reshape(-1, 2)
                audio_array = np.mean(audio_array, axis=1).astype(np.int16)
            
            # Resample if needed (simplified version)
            if params.framerate != 16000:
                ratio = 16000 / params.framerate
                new_length = int(len(audio_array) * ratio)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), new_length),
                    np.arange(len(audio_array)),
                    audio_array
                ).astype(np.int16)
            
            # Save as 16kHz mono 16-bit PCM
            with wave.open(output_path, 'wb') as outfile:
                outfile.setnchannels(1)
                outfile.setsampwidth(2)
                outfile.setframerate(16000)
                outfile.writeframes(audio_array.tobytes())
        
        return True
    except Exception as e:
        st.error(f"Audio conversion failed: {str(e)}")
        return False

# Function to process audio
def process_audio(audio_file_path):
    try:
        # Create a temporary converted file
        with tempfile.NamedTemporaryFile(suffix=".wav") as converted_file:
            if convert_to_compatible_wav(audio_file_path, converted_file.name):
                # Transcribe with Whisper
                result = whisper_model.transcribe(converted_file.name)
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
        return None
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# Audio input section
with st.sidebar:
    st.header("Audio Input")
    input_method = st.radio("Choose input method:", ("Record Audio", "Upload Audio File"))
    
    audio_file = None
    
    if input_method == "Record Audio":
        st.write("Please record in Hindi/Hinglish (minimum 5 seconds):")
        audio_bytes = audio_recorder(pause_threshold=5.0, sample_rate=16000)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                audio_file = tmp.name
    else:
        uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                audio_file = tmp.name
            st.audio(audio_file, format="audio/wav")

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
    st.info("Please record or upload an audio file")

st.markdown("""
### Instructions:
1. Record or upload an audio file (WAV, MP3, or OGG)
2. Audio should be in Hindi or Hinglish
3. Click "Process Audio"
4. View results in different tabs
5. Download any results as text files

Note: First run will take 2-3 minutes to download models.
""")
