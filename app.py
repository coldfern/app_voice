import streamlit as st
import os
import io
import time
import numpy as np
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline
from googletrans import Translator

# Set page config
st.set_page_config(page_title="Hindi/Hinglish Audio Analyzer", page_icon="🎤", layout="wide")

# Initialize components
translator = Translator()
recognizer = sr.Recognizer()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Helper functions
def save_uploaded_file(uploaded_file):
    """Save uploaded file to disk"""
    try:
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def convert_to_wav(audio_file):
    """Convert audio file to WAV format using pydub"""
    try:
        if audio_file.name.endswith('.mp3'):
            sound = AudioSegment.from_mp3(audio_file)
        elif audio_file.name.endswith('.wav'):
            sound = AudioSegment.from_wav(audio_file)
        else:
            sound = AudioSegment.from_file(audio_file)
        
        # Export as WAV
        wav_path = "temp/audio.wav"
        sound.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        st.error(f"Error converting audio: {e}")
        return None

def transcribe_audio(audio_file_path, language='hi-IN'):
    """Transcribe audio using SpeechRecognition"""
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language)
            return text
    except sr.UnknownValueError:
        st.warning("Google Speech Recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return None

def translate_text(text, src='hi', dest='en'):
    """Translate text using googletrans"""
    try:
        translation = translator.translate(text, src=src, dest=dest)
        return translation.text
    except Exception as e:
        st.error(f"Error in translation: {e}")
        return None

def summarize_text(text):
    """Summarize text using HuggingFace pipeline"""
    try:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return None

def analyze_sentiment(text):
    """Analyze sentiment of English text"""
    try:
        result = sentiment_analyzer(text)[0]
        return result['label'], result['score']
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return None, None

def record_audio(duration=5):
    """Record audio from microphone"""
    try:
        with sr.Microphone() as source:
            st.info(f"Recording for {duration} seconds... Speak now!")
            audio_data = recognizer.record(source, duration=duration)
            st.success("Recording complete!")
            
            # Save as WAV
            wav_path = "temp/recorded_audio.wav"
            with open(wav_path, "wb") as f:
                f.write(audio_data.get_wav_data())
            return wav_path
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None

# App layout
st.title("🎤 Hindi/Hinglish Audio Analyzer")
st.markdown("""
Record or upload audio in Hindi/Hinglish to get:
- Transcription
- English translation
- Summary
- Sentiment analysis
""")

# Create temp directory
os.makedirs("temp", exist_ok=True)

# Audio input options
input_method = st.radio("Select input method:", ("Record Audio", "Upload Audio"))

audio_path = None

if input_method == "Record Audio":
    duration = st.slider("Recording duration (seconds):", 3, 30, 5)
    if st.button("Start Recording"):
        audio_path = record_audio(duration)
        
else:  # Upload Audio
    uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg', 'm4a'])
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            audio_path = convert_to_wav(uploaded_file)

# Process audio if available
if audio_path:
    st.audio(audio_path)
    
    if st.button("Process Audio"):
        with st.spinner("Processing..."):
            # Transcription
            st.subheader("Transcription")
            transcription = transcribe_audio(audio_path)
            
            if transcription:
                st.success("Hindi Transcription:")
                st.write(transcription)
                
                # Translation
                st.subheader("English Translation")
                translation = translate_text(transcription)
                
                if translation:
                    st.success("English Translation:")
                    st.write(translation)
                    
                    # Summary
                    st.subheader("Summary")
                    summary = summarize_text(translation)
                    
                    if summary:
                        st.success("Summary:")
                        st.write(summary)
                    
                    # Sentiment Analysis
                    st.subheader("Sentiment Analysis")
                    sentiment, score = analyze_sentiment(translation)
                    
                    if sentiment:
                        st.success(f"Sentiment: {sentiment} (Confidence: {score:.2f})")
                        
                        # Visual indicator
                        if sentiment == "POSITIVE":
                            st.markdown("😊 Positive")
                        else:
                            st.markdown("😞 Negative")
                            
                        st.progress(score if sentiment == "POSITIVE" else 1-score)
            else:
                st.error("Failed to transcribe audio")

# Clean up
for filename in os.listdir("temp"):
    file_path = os.path.join("temp", filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        st.error(f"Error cleaning up files: {e}")

st.markdown("---")
st.markdown("### How to use:")
st.markdown("""
1. Choose to record audio or upload an existing file
2. Click the process button
3. View the transcription, translation, summary, and sentiment analysis
""")

st.markdown("### Notes:")
st.markdown("""
- For best results, use clear audio with minimal background noise
- The app works best with Hindi or Hinglish (Hindi-English mix) content
- Processing may take some time depending on audio length
""")
