import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import threading
import os
import torch
import whisper
from deep_translator import GoogleTranslator

# Load the Whisper model
model = whisper.load_model("small")

def save_audio(filename, data, samplerate):
    """Save recorded audio to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)
        wf.writeframes(data)

def record_audio(duration, samplerate=44100):
    """Record audio for a given duration."""
    st.session_state.recording = True
    buffer = []
    def callback(indata, frames, time, status):
        if status:
            print(status)
        buffer.append(indata.copy())
    
    with sd.InputStream(callback=callback, channels=1, samplerate=samplerate, dtype='int16'):
        while st.session_state.recording:
            sd.sleep(100)
    
    audio_data = np.concatenate(buffer, axis=0)
    audio_data = (audio_data * 32767).astype(np.int16)  # Convert to int16
    save_audio("recorded_audio.wav", audio_data.tobytes(), samplerate)
    st.session_state.recording = False

def stop_recording():
    """Stop the audio recording."""
    st.session_state.recording = False

def transcribe_audio(filename):
    """Transcribe the Hindi audio using Whisper."""
    result = model.transcribe(filename, language="hi")
    return result["text"]

def translate_text(text):
    """Translate Hindi text to English."""
    translator = GoogleTranslator(source='hi', target='en')
    return translator.translate(text)

# Streamlit UI
st.title("Hindi Audio Transcription & Translation App")

if "recording" not in st.session_state:
    st.session_state.recording = False

st.write("Record your Hindi speech and get its transcript along with an English translation.")

duration = st.slider("Select Maximum Recording Duration (in seconds)", 10, 1200, 60)

if st.button("Start Recording"):
    if not st.session_state.recording:
        st.session_state.recording = True
        threading.Thread(target=record_audio, args=(duration,)).start()
        st.success("Recording started. Click 'Stop Recording' to end.")

if st.button("Stop Recording"):
    stop_recording()
    st.success("Recording stopped.")

if os.path.exists("recorded_audio.wav"):
    st.audio("recorded_audio.wav", format="audio/wav")
    st.write("### Transcription")
    transcript = transcribe_audio("recorded_audio.wav")
    st.write(transcript)
    
    st.write("### English Translation")
    translation = translate_text(transcript)
    st.write(translation)

st.write("Note: The maximum recording duration is 20 minutes.")
