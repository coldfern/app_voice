import streamlit as st
import speech_recognition as sr
import tempfile
import os
from transformers import pipeline
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pydub import AudioSegment

st.set_page_config(page_title="Hindi Audio App", layout="centered")
st.title("🎙️ Hindi/Hinglish Audio Summarizer")

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="google/pegasus-xsum")
    return summarizer

summarizer = load_models()
translator = Translator()
analyzer = SentimentIntensityAnalyzer()

# Upload audio file
audio_file = st.file_uploader("Upload Hindi/Hinglish audio (wav/mp3)", type=["wav", "mp3"])

if audio_file:
    st.info("📥 Processing audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        if audio_file.type == "audio/mp3":
            sound = AudioSegment.from_file(audio_file, format="mp3")
            sound.export(temp_audio.name, format="wav")
        else:
            temp_audio.write(audio_file.read())
        temp_path = temp_audio.name

    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_path) as source:
        audio_data = recognizer.record(source)
        try:
            st.info("🗣️ Transcribing using Google Web Speech API...")
            text = recognizer.recognize_google(audio_data, language="hi-IN")
            st.success("✅ Transcription complete.")

            st.subheader("📜 Transcription (Hindi/Hinglish):")
            st.write(text)

            st.info("🌐 Translating to English...")
            translated = translator.translate(text, src="hi", dest="en").text
            st.subheader("🔤 Translation:")
            st.write(translated)

            st.info("🧠 Summarizing...")
            summary = summarizer(translated, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
            st.subheader("📝 Summary:")
            st.write(summary)

            st.info("📊 Analyzing Sentiment...")
            sentiment = analyzer.polarity_scores(translated)
            label = max(sentiment, key=sentiment.get).capitalize()
            st.subheader("💬 Sentiment:")
            st.write(f"{label} ({sentiment})")

        except Exception as e:
            st.error(f"Transcription failed: {e}")
    os.remove(temp_path)
