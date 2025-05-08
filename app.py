import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
from transformers import pipeline
import tempfile
import os

# Set up models
@st.cache_resource
def load_models():
    # Hindi ASR
    asr_pipe = pipeline("automatic-speech-recognition", 
                       model="Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
    
    # Translation & text processing
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment = pipeline("sentiment-analysis")
    
    return asr_pipe, translator, summarizer, sentiment

# Audio callback for recording
def audio_callback(frame):
    if not hasattr(audio_callback, "audio_frames"):
        audio_callback.audio_frames = []
    audio_callback.audio_frames.append(frame.to_ndarray())
    return frame

def main():
    st.title("🎙️ Audio Processor")
    st.write("Record Hindi audio to get transcription, translation, summary and sentiment analysis")

    # Audio recording with webrtc
    ctx = webrtc_streamer(
        key="audio-recorder",
        mode=av.AudioFrame,
        audio_callback=audio_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    if st.button("Process Recording") and hasattr(audio_callback, "audio_frames"):
        with st.spinner("Processing..."):
            # Combine audio frames
            audio_array = np.concatenate(audio_callback.audio_frames)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                import soundfile as sf
                sf.write(tmp_path, audio_array, 16000)
                
                # Load models
                asr_pipe, translator, summarizer, sentiment = load_models()
                
                # Transcription
                transcript = asr_pipe(tmp_path)["text"]
                st.subheader("Hindi Transcription")
                st.write(transcript)
                
                # Translation
                translation = translator(transcript)[0]["translation_text"]
                st.subheader("English Translation")
                st.write(translation)
                
                # Summary
                summary = summarizer(translation, max_length=130, min_length=30)[0]["summary_text"]
                st.subheader("Summary")
                st.write(summary)
                
                # Sentiment
                sentiment_result = sentiment(translation)[0]
                st.subheader("Sentiment Analysis")
                st.write(f"Label: {sentiment_result['label']}")
                st.write(f"Confidence: {sentiment_result['score']:.2f}")
            
            # Clean up
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
