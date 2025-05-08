import streamlit as st
import pyaudio
import wave
import numpy as np
import scipy.io.wavfile as wav
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from googletrans import Translator
import io
import librosa
import soundfile as sf

# Load pre-trained Wav2Vec2 model and processor for Hindi ASR
processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
model = Wav2Vec2ForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")

# Initialize translator
translator = Translator()

# Function to record audio
def record_audio(duration, sample_rate=16000):
    st.write("🎙️ Recording...")
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    st.write("✅ Recording complete.")
    return np.frombuffer(b''.join(frames), sample_rate

# Function to save audio as WAV
def save_audio(audio, sample_rate, filename="recorded_audio.wav"):
    wav.write(filename, sample_rate, audio)
    return filename

# Function to process and transcribe audio
def transcribe_audio(audio_data, sample_rate):
    # Convert audio data to float32 (normalize)
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

    # Preprocess for Wav2Vec2
    input_values = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_values.float()

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

    return transcription

# Function to translate Hindi text to English
def translate_to_english(text):
    translation = translator.translate(text, src='hi', dest='en')
    return translation.text

# Function to summarize English text
def summarize_text(text, max_length=250, min_length=30):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Streamlit App
def main():
    st.title("🎙️ Record Audio to Get Transcripts and Summary :)")

    # Option: Record or Upload
    option = st.radio("Choose an option:", ["🎤 Record Audio", "📂 Upload Audio File"])

    audio_data = None
    sample_rate = 16000  # Default sample rate

    if option == "🎤 Record Audio":
        duration = st.slider("Select recording duration (seconds)", 1, 1200, 10)
        if st.button("Start Recording"):
            audio, sample_rate = record_audio(duration)
            audio_path = save_audio(audio, sample_rate)
            st.audio(audio_path, format="audio/wav")
            audio_data, _ = librosa.load(audio_path, sr=sample_rate)

    elif option == "📂 Upload Audio File":
        uploaded_file = st.file_uploader("Upload an audio file...", type=["wav", "mp3", "ogg"])
        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")
            with io.BytesIO(uploaded_file.read()) as audio_io:
                audio_data, sample_rate = librosa.load(audio_io, sr=16000)

    # If audio is available, process it
    if audio_data is not None:
        st.write("📝 **Transcribing audio...**")
        transcription = transcribe_audio(audio_data, sample_rate)
        st.write("**Hindi Transcription:**")
        st.success(transcription)

        st.write("🌍 **Translating to English...**")
        translation = translate_to_english(transcription)
        st.write("**English Translation:**")
        st.info(translation)

        st.write("✍️ **Summarizing...**")
        summary = summarize_text(translation)
        st.write("**Summary:**")
        st.warning(summary)

if __name__ == "__main__":
    main()
