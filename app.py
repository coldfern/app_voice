import streamlit as st
import tempfile
import whisper
import argostranslate.package
import argostranslate.translate
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder
import os

# Page config
st.set_page_config(page_title="Hindi Audio AI", layout="centered")
st.title("🎙️ Hindi/Hinglish Audio Analyzer")
st.write("Upload or record Hindi/Hinglish audio and get a transcript, translation, summary, and sentiment.")

# Load models once
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment_analyzer = pipeline("sentiment-analysis")
    return whisper_model, summarizer, sentiment_analyzer

# Setup translation
@st.cache_resource
def setup_translation():
    packages = argostranslate.package.get_available_packages()
    hi_en = list(filter(lambda p: p.from_code == "hi" and p.to_code == "en", packages))[0]
    path = hi_en.download()
    argostranslate.package.install_from_path(path)
    return argostranslate.translate

# Audio input
choice = st.radio("Select Input Method:", ["Record Audio", "Upload File"])

tmp_audio_path = None

if choice == "Record Audio":
    audio = audio_recorder(pause_threshold=2.0)
    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio)
            tmp_audio_path = f.name
        st.audio(tmp_audio_path, format="audio/wav")
else:
    file = st.file_uploader("Upload audio file (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(file.read())
            tmp_audio_path = f.name
        st.audio(tmp_audio_path)

# If audio exists, process it
if tmp_audio_path:
    with st.spinner("Processing audio..."):
        whisper_model, summarizer, sentiment_analyzer = load_models()
        translator = setup_translation()

        result = whisper_model.transcribe(tmp_audio_path, language="hi")
        hindi_text = result["text"]

        st.subheader("📝 Hindi Transcript")
        st.write(hindi_text)

        english = translator.translate(hindi_text, "hi", "en")
        st.subheader("🔤 English Translation")
        st.write(english)

        summary = summarizer(english, max_length=100, min_length=25, do_sample=False)[0]['summary_text']
        st.subheader("🧾 Summary")
        st.write(summary)

        sentiment = sentiment_analyzer(english)[0]
        st.subheader("📊 Sentiment")
        st.write(f"**{sentiment['label']}** (confidence: {sentiment['score']:.2f})")

        os.remove(tmp_audio_path)
