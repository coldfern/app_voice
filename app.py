import streamlit as st
import tempfile
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder
import whisper
from googletrans import Translator

# App setup
st.set_page_config(page_title="Hindi Audio Processor", layout="centered")
st.title("🎙️ Hindi/Hinglish Audio Processor")

# Initialize models (cached)
@st.cache_resource
def load_models():
    try:
        # Load Whisper model
        model = whisper.load_model("base")  # Using 'base' for better accuracy
        
        # Initialize translator
        translator = Translator()
        
        # Load sentiment analysis pipeline
        sentiment_analyzer = pipeline("sentiment-analysis")
        
        return model, translator, sentiment_analyzer
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

# Load models when app starts
whisper_model, translator, sentiment_analyzer = load_models()

# Audio processing function
def process_audio(audio_path):
    try:
        # Transcribe with Whisper
        result = whisper_model.transcribe(audio_path)
        hindi_text = result["text"]
        
        # Translate to English
        english_text = translator.translate(hindi_text, src='hi', dest='en').text
        
        # Generate summary (first 2 sentences)
        sentences = [s.strip() for s in english_text.split('.') if s.strip()]
        summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else english_text
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer(english_text[:512])[0]
        
        return {
            "Hindi Transcript": hindi_text,
            "English Translation": english_text,
            "Summary": summary,
            "Sentiment": f"{sentiment_result['label']} ({sentiment_result['score']:.0%} confidence)"
        }
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# Main app interface
def main():
    # Audio input options
    input_method = st.radio("Select input method:", ("Record Audio", "Upload Audio File"))

    audio_data = None
    
    if input_method == "Record Audio":
        st.write("Speak in Hindi/Hinglish (minimum 5 seconds):")
        audio_data = audio_recorder(pause_threshold=5.0, sample_rate=16000)
        if audio_data:
            st.audio(audio_data, format="audio/wav")
    else:
        uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
        if uploaded_file:
            audio_data = uploaded_file.read()
            st.audio(audio_data, format="audio/wav")

    # Process audio when available
    if audio_data and st.button("Process Audio", type="primary"):
        with st.spinner("Processing your audio..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                if isinstance(audio_data, bytes):
                    tmp_file.write(audio_data)
                else:
                    tmp_file.write(audio_data.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process the audio file
            results = process_audio(tmp_file_path)
            
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            # Display results
            if results:
                st.subheader("Results")
                st.json(results)
                
                # Download button
                st.download_button(
                    "Download Results",
                    str(results),
                    file_name="audio_analysis_results.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    import os
    main()
