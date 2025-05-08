import streamlit as st
import whisper
import torchaudio
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import argostranslate.package
import argostranslate.translate

# Setup page
st.set_page_config(page_title="Hindi Audio App", layout="centered")
st.title("🎙️ Hindi/Hinglish Audio App (No ffmpeg)")

# Load Whisper and NLP models
@st.cache_resource
def load_models():
    model = whisper.load_model("base")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment = pipeline("sentiment-analysis")
    return model, summarizer, sentiment

@st.cache_resource
def setup_translation():
    packages = argostranslate.package.get_available_packages()
    hi_en = list(filter(lambda p: p.from_code == "hi" and p.to_code == "en", packages))[0]
    path = hi_en.download()
    argostranslate.package.install_from_path(path)
    return argostranslate.translate

# Input
input_method = st.radio("Choose input method:", ["Record Audio", "Upload WAV file"])
tmp_path = None

if input_method == "Record Audio":
    audio_bytes = audio_recorder(pause_threshold=2.0)
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            tmp_path = f.name
        st.audio(tmp_path, format="audio/wav")
else:
    file = st.file_uploader("Upload only .wav file", type=["wav"])
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(file.read())
            tmp_path = f.name
        st.audio(tmp_path, format="audio/wav")

# Process
if tmp_path:
    with st.spinner("Processing..."):
        model, summarizer, sentiment = load_models()
        translate = setup_translation()

        result = model.transcribe(tmp_path, language="hi")
        hindi_text = result["text"]
        st.subheader("📝 Hindi Transcript")
        st.write(hindi_text)

        english = translate.translate(hindi_text, "hi", "en")
        st.subheader("🔤 English Translation")
        st.write(english)

        summary = summarizer(english, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        st.subheader("🧾 Summary")
        st.write(summary)

        sentiment_result = sentiment(english)[0]
        st.subheader("📊 Sentiment")
        st.write(f"**{sentiment_result['label']}** with confidence **{sentiment_result['score']:.2f}**")

        os.remove(tmp_path)
