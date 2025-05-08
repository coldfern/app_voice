import streamlit as st
import os
import tempfile
from transformers import pipeline
import whisper
import warnings
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Suppress warnings
warnings.filterwarnings("ignore")

# App title and config
st.set_page_config(page_title="Hindi/Hinglish Audio Processor", layout="wide")
st.title("🇮🇳 Hindi/Hinglish Audio Processor")

# Initialize models (cache them for performance)
@st.cache_resource
def load_models():
    # Whisper model for transcription (small is faster than base for Hindi)
    whisper_model = whisper.load_model("small")
    
    # Translation pipeline (Helsinki-NLP Opus MT models)
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
    
    # Sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    
    return whisper_model, translator, sentiment_analyzer

whisper_model, translator, sentiment_analyzer = load_models()

# Function to process audio
def process_audio(audio_file_path):
    # Transcribe with Whisper
    result = whisper_model.transcribe(audio_file_path)
    hindi_text = result["text"]
    
    # Translate to English
    try:
        english_text = translator(hindi_text)[0]['translation_text']
    except:
        # Fallback to transliteration if translation fails
        english_text = transliterate(hindi_text, sanscript.DEVANAGARI, sanscript.ITRANS)
    
    # Generate summary (simple approach)
    sentences = english_text.split('. ')
    summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else english_text
    
    # Sentiment analysis
    try:
        sentiment_result = sentiment_analyzer(english_text[:512])[0]  # Limit to 512 tokens
        sentiment = f"{sentiment_result['label']} (confidence: {sentiment_result['score']:.2f})"
    except:
        sentiment = "Sentiment analysis failed"
    
    return {
        "hindi_transcript": hindi_text,
        "english_translation": english_text,
        "summary": summary,
        "sentiment": sentiment
    }

# HTML and JS for audio recording
audio_recorder_html = """
    <script>
        let mediaRecorder;
        let audioChunks = [];

        // Initialize recorder
        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks);
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audio = document.createElement('audio');
                        audio.controls = true;
                        audio.src = audioUrl;
                        document.getElementById('audioContainer').appendChild(audio);

                        const audioData = audioBlob;
                        const reader = new FileReader();
                        reader.onloadend = function () {
                            const base64Audio = reader.result.split(',')[1];
                            const audioDataElement = document.getElementById('audioData');
                            audioDataElement.value = base64Audio;
                        }
                        reader.readAsDataURL(audioBlob);
                    }
                    mediaRecorder.start();
                });
        }

        // Stop recording
        function stopRecording() {
            mediaRecorder.stop();
        }
    </script>
    <button onclick="startRecording()">Start Recording</button>
    <button onclick="stopRecording()">Stop Recording</button>
    <div id="audioContainer"></div>
    <input type="hidden" id="audioData">
"""

# Display HTML and JS in Streamlit
st.components.v1.html(audio_recorder_html, height=300)

# Sidebar for audio input
with st.sidebar:
    st.header("Audio Input Options")
    input_method = st.radio("Choose input method:", ("Record Audio", "Upload Audio File"))
    
    audio_file = None
    audio_data = st.text_input("Audio Data (Base64)")  # To get the base64 data from JS
    
    if input_method == "Upload Audio File":
        audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
        if audio_file:
            st.audio(audio_file, format="audio/wav")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                audio_file = tmp.name

# Main processing
if audio_data:
    # Convert base64 to binary
    import base64
    audio_bytes = base64.b64decode(audio_data)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        audio_file = tmp.name

if audio_file:
    if st.button("Process Audio"):
        with st.spinner("Processing your audio... This may take 1-2 minutes..."):
            try:
                results = process_audio(audio_file)
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs([ 
                    "Hindi Transcript", 
                    "English Translation", 
                    "Summary", 
                    "Sentiment Analysis"
                ])
                
                with tab1:
                    st.subheader("Hindi Transcript")
                    st.write(results["hindi_transcript"])
                    st.download_button(
                        label="Download Hindi Transcript",
                        data=results["hindi_transcript"],
                        file_name="hindi_transcript.txt",
                        mime="text/plain"
                    )
                
                with tab2:
                    st.subheader("English Translation")
                    st.write(results["english_translation"])
                    st.download_button(
                        label="Download English Translation",
                        data=results["english_translation"],
                        file_name="english_translation.txt",
                        mime="text/plain"
                    )
                
                with tab3:
                    st.subheader("Summary")
                    st.write(results["summary"])
                    st.download_button(
                        label="Download Summary",
                        data=results["summary"],
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                
                with tab4:
                    st.subheader("Sentiment Analysis")
                    st.write(results["sentiment"])
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
else:
    st.info("Please record or upload an audio file to get started.")

# Add some instructions
st.markdown("""
### Instructions:
1. Choose to either record audio or upload an audio file (preferably in Hindi or Hinglish)
2. Click "Process Audio" button
3. View the results in different tabs:
   - Hindi Transcript
   - English Translation
   - Summary
   - Sentiment Analysis
4. Download the results as text files

Note: First-time processing may take 2-3 minutes to download models.
""")
