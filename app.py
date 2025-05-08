import streamlit as st
import numpy as np
from transformers import pipeline
import librosa
import soundfile as sf
import tempfile
import os

# Set up models (will download on first run)
@st.cache_resource
def load_models():
    # Hindi ASR model
    asr_pipe = pipeline("automatic-speech-recognition", model="Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
    
    # Translation & text processing
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    return asr_pipe, translator, summarizer, sentiment

def main():
    st.title("🎙️ Audio Processor")
    st.write("Record or upload Hindi audio to get transcription, translation, summary and sentiment analysis")

    # Audio input
    audio_bytes = st.audio_recorder("Click to record", pause_threshold=2.0)
    
    if audio_bytes and audio_bytes['bytes']:
        # Save recording to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes['bytes'])
            audio_path = tmp.name
        
        st.audio(audio_bytes['bytes'], format="audio/wav")
        
        if st.button("Process Audio"):
            with st.spinner("Loading models..."):
                asr_pipe, translator, summarizer, sentiment = load_models()
            
            # Convert to 16kHz mono for ASR
            with st.spinner("Preparing audio..."):
                y, sr = librosa.load(audio_path, sr=16000, mono=True)
                sf.write(audio_path, y, sr)
            
            # Transcription
            with st.spinner("Transcribing..."):
                transcript = asr_pipe(audio_path)["text"]
                st.subheader("Hindi Transcription")
                st.write(transcript)
            
            # Translation
            with st.spinner("Translating..."):
                translation = translator(transcript)[0]["translation_text"]
                st.subheader("English Translation")
                st.write(translation)
            
            # Summary
            with st.spinner("Summarizing..."):
                summary = summarizer(translation, max_length=130, min_length=30)[0]["summary_text"]
                st.subheader("Summary")
                st.write(summary)
            
            # Sentiment
            with st.spinner("Analyzing sentiment..."):
                sentiment_result = sentiment(translation)[0]
                st.subheader("Sentiment Analysis")
                st.write(f"Label: {sentiment_result['label']}")
                st.write(f"Confidence: {sentiment_result['score']:.2f}")
        
        # Clean up
        os.unlink(audio_path)

if __name__ == "__main__":
    main()
