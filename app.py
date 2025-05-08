import streamlit as st
from streamlit_audiorecorder import audiorecorder
import speech_recognition as sr
from googletrans import Translator
from transformers import pipeline
import tempfile
import os

st.set_page_config(
    page_title="Hindi Audio Transcription, Translation & Analysis",
    layout="centered",
    initial_sidebar_state="auto"
)

# Inline CSS for modern styling and responsiveness
st.markdown(
    """
    <style>
    .main > .block-container {
        padding: 1rem 2rem;
        max-width: 700px;
        margin: auto;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
          Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
        color: #0e1117;
    }
    h1 {
        color: #4a90e2;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 0.2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg,#4a90e2 0%,#007aff 100%);
        color: white;
        font-weight: 600;
        border-radius: 0.4rem;
        padding: 0.4rem 1rem;
        transition: background 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg,#007aff 0%,#4a90e2 100%);
        color: white;
    }
    .section-header {
        margin-top: 2rem;
        font-weight: 600;
        border-bottom: 2px solid #4a90e2;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎤 Hindi / Hinglish Audio Transcriber & Analyzer")
st.write("Record or upload your Hindi/Hinglish audio. Get transcript, English translation, summary, and sentiment analysis.")

# Initialize Translator and NLP pipelines once to avoid reload penalties
@st.cache_resource
def load_translator():
    return Translator()

@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource(show_spinner=False)
def load_sentiment():
    return pipeline("sentiment-analysis")

translator = load_translator()
summarizer = load_summarizer()
sentiment_analyzer = load_sentiment()

def transcribe_audio(file_path):
    """Transcribe Hindi/Hinglish audio to text using Google's Web Speech API."""
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = r.record(source)
    try:
        # Language hints for Hindi - Google's recognizer supports 'hi-IN'
        text = r.recognize_google(audio_data, language="hi-IN")
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

def translate_text(text):
    try:
        # Translate detected text to English
        translation = translator.translate(text, src='hi', dest='en')
        return translation.text
    except Exception:
        return None

def summarize_text(text):
    try:
        # Huggingface pipeline summarization limits max tokens input, so chunk if needed
        max_chunk = 500
        # Chunk text intelligently
        if len(text) > max_chunk:
            chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
            summaries = [summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]['summary_text'] for chunk in chunks]
            return " ".join(summaries)
        else:
            summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
            return summary
    except Exception:
        return None

def analyze_sentiment(text):
    try:
        result = sentiment_analyzer(text)
        # result is a list of dicts with 'label' and 'score'
        return result[0]['label'], result[0]['score']
    except Exception:
        return None, None

# Tabs for recording or uploading audio
tab1, tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload Audio"])

transcript = None
translated = None
summary = None
sentiment = None

with tab1:
    st.markdown("### Record your audio (max 30 seconds)")
    audio_bytes = audiorecorder("Start/Stop Recording", recording_color="#4a90e2", neutral_color="#6c757d", sample_rate=16000)
    if len(audio_bytes) > 0:
        st.audio(audio_bytes, format="audio/wav")
        # Save to a temporary wav file to process
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(audio_bytes)
            tmp_wav_path = tmp_wav.name

        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(tmp_wav_path)
        os.unlink(tmp_wav_path)

        if transcript:
            st.success("Transcription successful!")
            st.markdown(f"**Transcript (Hindi/Hinglish):**  \n{transcript}")
        else:
            st.error("Sorry, could not transcribe the audio. Please try again.")

with tab2:
    st.markdown("### Upload your audio file (WAV format only)")
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(uploaded_file.read())
            tmp_wav_path = tmp_wav.name

        with st.spinner("Transcribing uploaded audio..."):
            transcript = transcribe_audio(tmp_wav_path)

        os.unlink(tmp_wav_path)

        if transcript:
            st.success("Transcription successful!")
            st.markdown(f"**Transcript (Hindi/Hinglish):**  \n{transcript}")
        else:
            st.error("Sorry, could not transcribe the uploaded audio. Please upload a clear WAV audio.")

# If transcript obtained, process translation, summary, sentiment
if transcript:
    with st.expander("🇬🇧 English Translation"):
        with st.spinner("Translating transcript..."):
            translated = translate_text(transcript)
        if translated:
            st.markdown(translated)
        else:
            st.error("Translation failed.")

    if translated:
        with st.expander("📝 Summary"):
            with st.spinner("Generating summary..."):
                summary = summarize_text(translated)
            if summary:
                st.markdown(summary)
            else:
                st.error("Summary generation failed.")

        with st.expander("😊 Sentiment Analysis"):
            with st.spinner("Analyzing sentiment..."):
                sentiment_label, sentiment_score = analyze_sentiment(summary if summary else translated)
            if sentiment_label:
                sentiment_color = "#4CAF50" if sentiment_label.lower() in ["positive", "joy", "happy"] else "#F44336" if sentiment_label.lower() in ["negative", "sadness", "anger", "fear"] else "#FFA500"
                st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}; font-weight:bold'>{sentiment_label}</span> (Confidence: {sentiment_score:.2f})", unsafe_allow_html=True)
            else:
                st.error("Sentiment analysis failed.")
else:
    st.info("Record or upload an audio file above to get started.")

st.markdown(
    """
    <hr>
    <footer style="text-align:center; font-size:0.8rem; color: #777;">
    Developed with ❤️ using Streamlit. Supports Hindi and Hinglish audio transcription.
    </footer>
    """,
    unsafe_allow_html=True,
)


