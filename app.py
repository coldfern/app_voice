import streamlit as st
import whisper
import tempfile
import os
from transformers import pipeline
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Hindi Audio Summarizer", layout="centered")
st.title("🎙️ Hindi/Hinglish Audio to English Transcript, Summary & Sentiment")

# Load resources
@st.cache_resource
def load_models():
    model = whisper.load_model("base")
    summarizer = pipeline("summarization", model="google/pegasus-xsum")
    return model, summarizer

model, summarizer = load_models()
translator = Translator()
analyzer = SentimentIntensityAnalyzer()

# Upload or record
audio_file = st.file_uploader("Upload your Hindi/Hinglish audio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(audio_file.read())
        temp_path = temp_audio.name

    st.info("🔁 Transcribing audio using Whisper...")
    result = model.transcribe(temp_path)
    hindi_text = result["text"]
    st.success("✅ Transcription complete.")

    st.subheader("📜 Transcription (Hindi/Hinglish):")
    st.write(hindi_text)

    st.info("🌐 Translating to English...")
    translated = translator.translate(hindi_text, src="hi", dest="en")
    english_text = translated.text
    st.subheader("🔤 English Translation:")
    st.write(english_text)

    st.info("🧠 Generating Summary...")
    summary = summarizer(english_text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    st.subheader("📝 Summary:")
    st.write(summary)

    st.info("💬 Performing Sentiment Analysis...")
    sentiment = analyzer.polarity_scores(english_text)
    sentiment_label = max(sentiment, key=sentiment.get).capitalize()
    st.subheader("📈 Sentiment:")
    st.write(f"{sentiment_label} ({sentiment})")

    # Clean up temp file
    os.remove(temp_path)
else:
    st.info("Please upload a .mp3, .wav or .m4a file to get started.")
