import streamlit as st
import os
import tempfile
import whisper
import argostranslate.package
import argostranslate.translate
from transformers import pipeline

# Load whisper model
st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Load translation model (Hindi -> English)
@st.cache_resource
def setup_translation():
    import os
    import subprocess
    packages = argostranslate.package.get_available_packages()
    package_to_install = list(
        filter(lambda x: x.from_code == "hi" and x.to_code == "en", packages)
    )[0]
    download_path = package_to_install.download()
    argostranslate.package.install_from_path(download_path)
    return argostranslate.translate

# Load summarizer and sentiment analyzer
@st.cache_resource
def load_nlp_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment_analyzer = pipeline("sentiment-analysis")
    return summarizer, sentiment_analyzer

st.title("🎙️ Hindi Audio Transcriber & Analyzer")
st.markdown("""
Upload or record Hindi/Hinglish audio. We'll transcribe it, translate it to English, summarize the content, and analyze the sentiment — all in one place!
""")

# Upload or record audio
audio_file = st.file_uploader("Upload an audio file (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/mp3")

    # Load models
    st.info("⏳ Loading models...")
    whisper_model = load_whisper_model()
    translate = setup_translation()
    summarizer, sentiment_analyzer = load_nlp_models()

    # Transcribe
    st.info("🔠 Transcribing...")
    result = whisper_model.transcribe(tmp_path, language="hi")
    hindi_text = result["text"]
    st.subheader("📝 Hindi Transcript")
    st.write(hindi_text)

    # Translate
    st.info("🌐 Translating to English...")
    translated_text = translate.translate(hindi_text, "hi", "en")
    st.subheader("🔤 English Translation")
    st.write(translated_text)

    # Summarize
    st.info("📄 Summarizing...")
    summary = summarizer(translated_text, max_length=100, min_length=25, do_sample=False)[0]['summary_text']
    st.subheader("🧾 Summary")
    st.write(summary)

    # Sentiment
    st.info("💬 Analyzing Sentiment...")
    sentiment = sentiment_analyzer(translated_text)[0]
    st.subheader("📊 Sentiment")
    st.write(f"**Label:** {sentiment['label']} | **Confidence:** {sentiment['score']:.2f}")

    # Cleanup
    os.remove(tmp_path)
