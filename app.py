import streamlit as st
import os
import tempfile
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import torch
from faster_whisper import WhisperModel
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# App title and config
st.set_page_config(page_title="Hindi/Hinglish Audio Processor", layout="wide")
st.title("🇮🇳 Hindi/Hinglish Audio Processor")

# Initialize models (cache them for performance)
@st.cache_resource
def load_models():
    # Faster Whisper model for transcription (tiny is fastest)
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    # Translation pipeline (smaller model)
    translator = pipeline("translation_hi_to_en", 
                         model="Helsinki-NLP/opus-mt-hi-en",
                         device="cpu")
    
    # Sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="distilbert-base-uncased-finetuned-sst-2-english",
                                device="cpu")
    
    return whisper_model, translator, sentiment_analyzer

try:
    whisper_model, translator, sentiment_analyzer = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Function to process audio
def process_audio(audio_file_path):
    # Transcribe with Whisper
    segments, info = whisper_model.transcribe(audio_file_path)
    hindi_text = " ".join([segment.text for segment in segments])
    
    # Translate to English
    try:
        english_text = translator(hindi_text)[0]['translation_text']
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        english_text = "Could not translate. Original text: " + hindi_text
    
    # Generate summary (simple approach)
    sentences = english_text.split('. ')
    summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else english_text
    
    # Sentiment analysis
    try:
        sentiment_result = sentiment_analyzer(english_text[:512])[0]  # Limit to 512 tokens
        sentiment = f"{sentiment_result['label']} (confidence: {sentiment_result['score']:.2f})"
    except:
        sentiment = "Sentiment analysis failed"
    
    return {
        "hindi_transcript": hindi_text,
        "english_translation": english_text,
        "summary": summary,
        "sentiment": sentiment
    }

# Sidebar for audio input
with st.sidebar:
    st.header("Audio Input Options")
    input_method = st.radio("Choose input method:", ("Record Audio", "Upload Audio File"))
    
    audio_data = None
    audio_file = None
    
    if input_method == "Record Audio":
        st.write("Record your Hindi/Hinglish audio (5 seconds minimum):")
        audio_bytes = audio_recorder(pause_threshold=5.0, sample_rate=16_000)
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                audio_file = tmp.name
    
    else:  # Upload Audio File
        audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
        if audio_file:
            st.audio(audio_file, format="audio/wav")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                audio_file = tmp.name

# Main processing
if audio_file:
    if st.button("Process Audio"):
        with st.spinner("Processing your audio... This may take 1-2 minutes..."):
            try:
                results = process_audio(audio_file)
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Hindi Transcript", 
                    "English Translation", 
                    "Summary", 
                    "Sentiment Analysis"
                ])
                
                with tab1:
                    st.subheader("Hindi Transcript")
                    st.write(results["hindi_transcript"])
                    st.download_button(
                        label="Download Hindi Transcript",
                        data=results["hindi_transcript"],
                        file_name="hindi_transcript.txt",
                        mime="text/plain"
                    )
                
                with tab2:
                    st.subheader("English Translation")
                    st.write(results["english_translation"])
                    st.download_button(
                        label="Download English Translation",
                        data=results["english_translation"],
                        file_name="english_translation.txt",
                        mime="text/plain"
                    )
                
                with tab3:
                    st.subheader("Summary")
                    st.write(results["summary"])
                    st.download_button(
                        label="Download Summary",
                        data=results["summary"],
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                
                with tab4:
                    st.subheader("Sentiment Analysis")
                    st.write(results["sentiment"])
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
else:
    st.info("Please record or upload an audio file to get started.")

# Add some instructions
st.markdown("""
### Instructions:
1. Choose to either record audio or upload an audio file (preferably in Hindi or Hinglish)
2. Click "Process Audio" button
3. View the results in different tabs:
   - Hindi Transcript
   - English Translation
   - Summary
   - Sentiment Analysis
4. Download the results as text files

Note: First-time processing may take 2-3 minutes to download models.
""")
