import streamlit as st
import tempfile
from transformers import pipeline
import whisper
from googletrans import Translator

# App setup
st.set_page_config(page_title="Hindi Audio Processor", layout="centered")
st.title("🎙️ Hindi/Hinglish Audio Processor")

# Initialize models (cached)
@st.cache_resource
def load_models():
    # Whisper for transcription
    transcribe_model = whisper.load_model("small")
    
    # Translation
    translator = Translator()
    
    # Sentiment analysis
    sentiment = pipeline("sentiment-analysis")
    
    return transcribe_model, translator, sentiment

# Audio processing function
def process_audio(audio_path):
    # Transcribe with Whisper
    result = transcribe_model.transcribe(audio_path)
    hindi_text = result["text"]
    
    # Translate to English
    english_text = translator.translate(hindi_text, src='hi', dest='en').text
    
    # Generate summary (first 2 sentences)
    summary = '. '.join(english_text.split('. ')[:2]) + '.'
    
    # Sentiment analysis
    sentiment_result = sentiment(english_text[:512])[0]
    
    return {
        "Hindi Transcript": hindi_text,
        "English Translation": english_text,
        "Summary": summary,
        "Sentiment": f"{sentiment_result['label']} ({sentiment_result['score']:.0%} confidence)"
    }

# Load models
transcribe_model, translator, sentiment = load_models()

# Audio input
audio_file = st.file_uploader("Upload Hindi audio", type=["wav", "mp3", "ogg"])
recorded_audio = audio_recorder("Or record audio", pause_threshold=5.0)

# Process when audio is available
if audio_file or recorded_audio:
    if st.button("Process Audio"):
        with st.spinner("Processing..."):
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                if audio_file:
                    tmp.write(audio_file.read())
                else:
                    tmp.write(recorded_audio)
                
                # Process the audio
                results = process_audio(tmp.name)
                
                # Display results
                st.subheader("Results")
                st.json(results)
                
                # Download button
                st.download_button(
                    "Download Results",
                    str(results),
                    file_name="audio_results.json"
                )
