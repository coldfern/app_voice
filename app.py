import streamlit as st
import speech_recognition as sr
from googletrans import Translator
from transformers import pipeline
import os
import tempfile
from pydub import AudioSegment
import numpy as np

# Set page config
st.set_page_config(page_title="Hindi/Hinglish Audio Processor", layout="wide")

# Initialize components
translator = Translator()
sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

def convert_to_wav(input_path, output_path):
    """Convert any audio file to WAV format"""
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")
    return output_path

def transcribe_audio(audio_path, language='hi-IN'):
    """Transcribe audio using SpeechRecognition"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "API unavailable"

def translate_text(text, src='hi', dest='en'):
    """Translate text using googletrans"""
    try:
        translation = translator.translate(text, src=src, dest=dest)
        return translation.text
    except:
        return "Translation failed"

def analyze_sentiment(text):
    """Analyze sentiment of English text"""
    try:
        result = sentiment_analyzer(text)[0]
        return f"{result['label']} (confidence: {result['score']:.2f})"
    except:
        return "Sentiment analysis failed"

def summarize_text(text):
    """Generate summary of English text"""
    try:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except:
        return "Summary generation failed"

def main():
    st.title("Hindi/Hinglish Audio Processor")
    st.write("Upload an audio file or record your voice to get transcriptions, translations, summaries and sentiment analysis.")

    # Audio input options
    input_method = st.radio("Choose input method:", ("Upload Audio", "Record Audio"))

    audio_path = None
    
    if input_method == "Upload Audio":
        uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg', 'm4a'])
        if uploaded_file:
            audio_path = save_uploaded_file(uploaded_file)
            if not audio_path.endswith('.wav'):
                with st.spinner("Converting to WAV format..."):
                    wav_path = audio_path + '.wav'
                    convert_to_wav(audio_path, wav_path)
                    os.unlink(audio_path)
                    audio_path = wav_path
            st.audio(audio_path)
    else:
        recorded_audio = st.audio_recorder("Click to record", pause_threshold=2.0)
        if recorded_audio:
            audio_bytes = recorded_audio['bytes']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                audio_path = tmp_file.name
            st.audio(audio_path)

    if audio_path and st.button("Process Audio"):
        with st.spinner("Processing..."):
            # Step 1: Transcribe
            st.subheader("Original Transcription")
            hindi_text = transcribe_audio(audio_path)
            st.write(hindi_text)

            if hindi_text and not hindi_text.startswith("Could not"):
                # Step 2: Translate
                st.subheader("English Translation")
                english_text = translate_text(hindi_text)
                st.write(english_text)

                # Step 3: Sentiment Analysis
                st.subheader("Sentiment Analysis")
                sentiment = analyze_sentiment(english_text)
                st.write(sentiment)

                # Step 4: Summary
                st.subheader("Summary")
                summary = summarize_text(english_text)
                st.write(summary)
            
        # Clean up temporary files
        if os.path.exists(audio_path):
            os.unlink(audio_path)

if __name__ == "__main__":
    main()
