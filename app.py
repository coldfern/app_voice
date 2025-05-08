import streamlit as st
import whisper
import tempfile
import os
from transformers import pipeline
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load models once
@st.cache_resource
def load_models():
    model = whisper.load_model("base")
    summarizer = pipeline("summarization")
    return model, summarizer

model, summarizer = load_models()
translator = Translator()
analyzer = SentimentIntensityAnalyzer()

st.title("🗣️ Hindi/Hinglish Audio to English Transcript, Summary & Sentiment")

# Upload or Record
st.header("Upload or Record Audio")
audio_file = st.file_uploader("Upload audio file (mp3/wav/m4a)", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        temp_audio_path = tmp_file.name

    with st.spinner("Transcribing audio..."):
        result = model.transcribe(temp_audio_path, language="hi")
        hindi_text = result["text"]

    st.subheader("📜 Original Transcription (Hindi/Hinglish)")
    st.write(hindi_text)

    with st.spinner("Translating to English..."):
        translated = translator.translate(hindi_text, src="hi", dest="en").text
    st.subheader("🌐 Translated Text (English)")
    st.write(translated)

    with st.spinner("Summarizing..."):
        summary = summarizer(translated, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    st.subheader("📝 Summary")
    st.write(summary)

    sentiment = analyzer.polarity_scores(translated)
    st.subheader("😊 Sentiment Analysis")
    st.write(f"Positive: {sentiment['pos']}, Neutral: {sentiment['neu']}, Negative: {sentiment['neg']}")
    st.write(f"Overall: {'Positive' if sentiment['compound'] > 0.05 else 'Negative' if sentiment['compound'] < -0.05 else 'Neutral'}")

    os.remove(temp_audio_path)
else:
    st.info("Please upload an audio file to continue.")

st.caption("Built with Whisper, Googletrans, Transformers & VADER")
