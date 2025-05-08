import streamlit as st
import os
import tempfile
import whisper
import argostranslate.package
import argostranslate.translate
from transformers import pipeline
from pathlib import Path
import base64

# Load whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Load translation model (Hindi -> English)
@st.cache_resource
def setup_translation():
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

st.set_page_config(page_title="Hindi Audio AI", layout="centered")
st.title("🎙️ Hindi Audio Transcriber & Analyzer")
st.markdown("""
Record or upload Hindi/Hinglish audio. Get a transcript, translation, summary, and sentiment — all for free!
""")

# Option to upload or record
input_mode = st.radio("Choose audio input method:", ["Upload", "Record"])

audio_file = None
tmp_path = None

if input_mode == "Upload":
    audio_file = st.file_uploader("Upload an audio file (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])
else:
    st.markdown("### 🔴 Record Audio")
    st.markdown(
        """
        <audio id="audio" controls></audio>
        <br>
        <button onclick="startRecording()">Start Recording</button>
        <button onclick="stopRecording()">Stop & Upload</button>
        <script>
        let mediaRecorder;
        let audioChunks = [];

        async function startRecording() {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks);
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    const base64Audio = reader.result.split(',')[1];
                    const pyCmd = `window.parent.postMessage({ type: 'streamlit:setComponentValue', value: "${base64Audio}" }, '*')`;
                    eval(pyCmd);
                };
            };
        }

        function stopRecording() {
            mediaRecorder.stop();
        }
        </script>
        """,
        unsafe_allow_html=True
    )

    base64_audio = st.query_params().get("audio_data", [None])[0]

    if base64_audio:
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(tmp_path, "wb") as f:
            f.write(base64.b64decode(base64_audio))
        st.audio(tmp_path)

# Process if audio is available
if audio_file or tmp_path:
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        st.audio(tmp_path, format="audio/mp3")

    st.info("⏳ Loading models...")
    whisper_model = load_whisper_model()
    translate = setup_translation()
    summarizer, sentiment_analyzer = load_nlp_models()

    st.info("🔠 Transcribing...")
    result = whisper_model.transcribe(tmp_path, language="hi")
    hindi_text = result["text"]
    st.subheader("📝 Hindi Transcript")
    st.write(hindi_text)

    st.info("🌐 Translating...")
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
