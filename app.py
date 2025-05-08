import streamlit as st
import os
import tempfile
import whisper
import argostranslate.package
import argostranslate.translate
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Setup translation
@st.cache_resource
def setup_translation():
    packages = argostranslate.package.get_available_packages()
    package_to_install = list(
        filter(lambda x: x.from_code == "hi" and x.to_code == "en", packages)
    )[0]
    download_path = package_to_install.download()
    argostranslate.package.install_from_path(download_path)
    return argostranslate.translate

# Load NLP models
@st.cache_resource
def load_nlp_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment_analyzer = pipeline("sentiment-analysis")
    return summarizer, sentiment_analyzer

# UI layout
st.set_page_config(page_title="Hindi Audio Analyzer", layout="centered")
st.title("🎙️ Hindi Audio Transcriber & Analyzer")
st.markdown("""
Record or upload Hindi/Hinglish audio. Get a transcript, translation, summary, and sentiment — all in one place.
""")

# Audio input option
input_mode = st.radio("Choose audio input method:", ["Upload", "Record"])

audio_file = None
tmp_path = None

if input_mode == "Upload":
    audio_file = st.file_uploader("Upload an audio file (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        st.audio(tmp_path, format="audio/mp3")

else:
    st.subheader("🎤 Record Audio")
    audio_bytes = audio_recorder(pause_threshold=2.0)
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            tmp_path = f.name
        st.audio(tmp_path, format="audio/wav")

# Process the audio
if tmp_path:
    st.info("⏳ Loading models...")
    whisper_model = load_whisper_model()
    translate = setup_translation()
    summarizer, sentiment_analyzer = load_nlp_models()

    st.info("🔠 Transcribing...")
    result = whisper_model.transcribe(tmp_path, language="hi")
    hindi_text = result["text"]
    st.subheader("📝 Hindi Transcript")
    st.write(hindi_text)

    st.info("🌐 Translating to English...")
    translated_text = translate.translate(hindi_text, "hi", "en")
    st.subheader("🔤 English Translation")
    st.write(translated_text)

    st.info("📄 Summarizing...")
    summary = summarizer(translated_text, max_length=100, min_length=25, do_sample=False)[0]['summary_text']
    st.subheader("🧾 Summary")
    st.write(summary)

    st.info("💬 Sentiment Analysis...")
    sentiment = sentiment_analyzer(translated_text)[0]
    st.subheader("📊 Sentiment")
    st.write(f"**Label:** {sentiment['label']} | **Confidence:** {sentiment['score']:.2f}")

    os.remove(tmp_path)
