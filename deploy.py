import os
import torch
import whisper
import torchaudio
from pydub import AudioSegment
from deep_translator import GoogleTranslator, exceptions
from denoiser import pretrained
from langdetect import detect
import re
import contractions
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
import nltk
import streamlit as st
import tempfile
import concurrent.futures

nltk.download("stopwords")
nltk.download("wordnet")

# Secure token handling
hf_token = os.getenv("HF_TOKEN") or st.secrets["HF_TOKEN"]

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face Model Hub paths
MENTAL_PATH = "sai1908/finetuned-dialoGPT-mental-health-llm-v1"
SENTIMENT_PATH = "sai1908/sentiment-analysis"
EMOTION_PATH = "sai1908/emotion-detection"
DISTILGPT_PATH = "sai1908/distilbert-mental-health-status-classifier"

@st.cache_resource
def load_models():
    def load_text_classification(path):
        tokenizer = AutoTokenizer.from_pretrained(path, use_auth_token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(path, use_auth_token=hf_token)
        return pipeline("text-classification", model=model, tokenizer=tokenizer)

    def load_emotion(path):
        tokenizer = AutoTokenizer.from_pretrained(path, use_auth_token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(path, use_auth_token=hf_token)
        return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    def load_gpt(path):
        tokenizer = AutoTokenizer.from_pretrained(path, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(path, use_auth_token=hf_token).to(device)
        return tokenizer, model

    def load_whisper():
        return whisper.load_model("base").to(device)

    def load_denoiser():
        return pretrained.dns64().to(device)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            "mental": executor.submit(load_text_classification, MENTAL_PATH),
            "sentiment": executor.submit(load_text_classification, SENTIMENT_PATH),
            "emotion": executor.submit(load_emotion, EMOTION_PATH),
            "chatbot": executor.submit(load_gpt, DISTILGPT_PATH),
            "whisper": executor.submit(load_whisper),
            "denoiser": executor.submit(load_denoiser)
        }
        results = {key: f.result() for key, f in futures.items()}

    return results

models = load_models()
mental_health_classifier = models["mental"]
sentiment_analyzer = models["sentiment"]
emotion_detector = models["emotion"]
chatbot_tokenizer, chatbot_model = models["chatbot"]
whisper_model = models["whisper"]
denoiser_model = models["denoiser"]

# Text preprocessing
def clean_text(text):
    text = contractions.fix(text).lower()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Analyzers

def analyze_sentiment(text):
    try:
        sentiment = sentiment_analyzer(text)[0]
        return {"label": sentiment["label"], "score": sentiment["score"]}
    except Exception as e:
        return {"error": str(e)}

def detect_emotions(text):
    try:
        emotions = emotion_detector(text)
        return [{"emotion": e["label"], "score": e["score"]} for e in emotions[0]]
    except Exception as e:
        return {"error": str(e)}

def classify_mental_health(text):
    try:
        result = mental_health_classifier(text)[0]
        return {"label": result["label"], "score": result["score"]}
    except Exception as e:
        return {"error": str(e)}

# Chatbot response

def generate_response(user_input, context):
    prompt = (
        f"User said: \"{user_input}\"\n"
        f"Sentiment: {context['Sentiment'].get('label', 'Unknown')}\n"
        f"Emotions: {', '.join([e['emotion'] for e in context['Emotions']])}\n"
        f"Mental Health: {context['Mental Health Classification'].get('label', 'Unknown')}\n"
        f"Generate a compassionate and empathetic response, providing support for the user's feelings."
    )
    inputs = chatbot_tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = chatbot_model.generate(
        **inputs, max_length=150, pad_token_id=chatbot_tokenizer.eos_token_id, do_sample=True, top_k=50, top_p=0.95
    )
    response = chatbot_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.split("feelings.")[-1].strip()

# Audio utilities

def preprocess_audio_with_format(audio_path, target_sr=16000):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    temp_wav_path = "temp_audio.wav"
    audio.export(temp_wav_path, format="wav")
    waveform, sr = torchaudio.load(temp_wav_path)
    waveform = waveform.to(device)
    with torch.no_grad():
        denoised_waveform = denoiser_model(waveform.unsqueeze(0)).squeeze(0)
    return denoised_waveform.cpu(), target_sr

def transcribe_audio(waveform):
    waveform_np = waveform.squeeze(0).numpy()
    result = whisper_model.transcribe(waveform_np)
    return result["text"]

def safe_translate(text):
    try:
        return GoogleTranslator(source=detect(text), target="en").translate(text)
    except exceptions.NotValidPayload:
        return text

# UI
st.title("🧠 Mental Health Support Chatbot")
tabs = st.tabs(["💬 Text Input", "🎙️ Audio Input"])

with tabs[0]:
    user_text = st.text_area("Enter your thoughts or feelings:", height=150)
    if st.button("Analyze Text"):
        if user_text.strip():
            with st.spinner("Analyzing your message..."):
                translated_text = safe_translate(user_text)
                cleaned_text = clean_text(translated_text)
                context = {
                    "Sentiment": analyze_sentiment(cleaned_text),
                    "Emotions": detect_emotions(cleaned_text),
                    "Mental Health Classification": classify_mental_health(cleaned_text)
                }
                chatbot_response = generate_response(cleaned_text, context)
                st.subheader("Context Analysis")
                st.json(context)
                st.subheader("Chatbot Response")
                st.success(chatbot_response)
        else:
            st.warning("Please enter some text.")

with tabs[1]:
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file and st.button("Analyze Audio"):
        with st.spinner("Processing your audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            preprocessed_audio, _ = preprocess_audio_with_format(tmp_path)
            transcription = transcribe_audio(preprocessed_audio)
            translated_text = safe_translate(transcription)
            cleaned_text = clean_text(translated_text)
            context = {
                "Sentiment": analyze_sentiment(cleaned_text),
                "Emotions": detect_emotions(cleaned_text),
                "Mental Health Classification": classify_mental_health(cleaned_text)
            }
            chatbot_response = generate_response(cleaned_text, context)
            st.subheader("Transcription")
            st.info(transcription)
            st.subheader("Context Analysis")
            st.json(context)
            st.subheader("Chatbot Response")
            st.success(chatbot_response)
