import streamlit as st
import speech_recognition as sr
import tempfile
import os
from transformers import pipeline
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pydub import AudioSegment
from audiorecorder import audiorecorder

st.set_page_config(page_title="Hindi Audio App", layout="centered")
st.title("🎙️ Hindi/Hinglish Audio Summarizer")

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="google/pegasus-xsum")
    return summarizer

summarizer = load_models()
translator = Translator()
analyzer = SentimentIntensityAnalyzer()

st.header("🎛️ Record or Upload Audio")

audio_bytes = audiorecorder("Click to record", "Recording...")

uploaded_file = st.file_uploader("Or upload an audio file (mp3/wav)", type=["mp3", "wav"])

audio_path = None

if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        audio_path = f.name
elif uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        if uploaded_file.type == "audio/mp3":
            sound = AudioSegment.from_file(uploaded_file, format="mp3")
            sound.export(f.name, format="wav")
        else:
            f.write(uploaded_file.read())
        audio_path = f.name

if audio_path:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            st.info("🗣️ Transcribing using Google Web Speech API...")
            text = recognizer.recognize_google(audio_data, language="hi-IN")
            st.success("✅ Transcription complete.")

            st.subheader("📜 Transcription (Hindi/Hinglish):")
            st.write(text)

            st.info("🌐 Translating to English...")
            translation = translator.translate(text, src="hi", dest="en").text
            st.subheader("🔤 Translation:")
            st.write(translation)

            st.info("🧠 Summarizing...")
            summary = summarizer(translation, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
            st.subheader("📝 Summary:")
            st.write(summary)

            st.info("📊 Analyzing Sentiment...")
            sentiment = analyzer.polarity_scores(translation)
            sentiment_label = max(sentiment, key=sentiment.get).capitalize()
            st.subheader("💬 Sentiment:")
            st.write(f"{sentiment_label} ({sentiment})")

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
    os.remove(audio_path)
else:
    st.info("Please record or upload an audio file to begin.")
