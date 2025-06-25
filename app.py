import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from PyPDF2 import PdfReader
import av
import whisper
from transformers import pipeline
import textwrap
import base64
from faker import Faker
from streamlit_player import st_player
from PIL import Image

# ---------- ENHANCED THEME & INIT ----------
st.set_page_config(
    page_title="AI Media Scheduler Pro",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
        :root {
            --primary: #6a11cb;
            --secondary: #2575fc;
            --accent: #ff6b6b;
            --dark: #2d3436;
            --light: #f7f9fc;
        }
        
        .stApp {
            background: linear-gradient(135deg, var(--light) 0%, #e6f7ff 100%);
            color: var(--dark);
        }
        
        .stButton>button {
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            border-radius: 25px;
            font-weight: bold;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        }
        
        .stHeader {
            color: var(--primary);
            text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ---------- AI INITIALIZATION ----------
@st.cache_resource
def load_ai_models():
    """Load all AI models at startup"""
    return {
        'whisper': whisper.load_model("small"),
        'summarizer': pipeline("summarization", model="facebook/bart-large-cnn"),
        'sentiment': pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"),
        'qa': pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    }

ai_models = load_ai_models()

# ---------- USER AUTHENTICATION ----------
USERS = {"chefking": "password123", "admin": "adminpass", "judge": "ai4good"}

def login():
    st.sidebar.subheader("🔐 Secure Login")
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login", use_container_width=True):
            if USERS.get(username) == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}! 👋")
            else:
                st.error("Invalid credentials")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------- INITIALIZE SESSION STATE ----------
if 'media_files' not in st.session_state:
    st.session_state.media_files = {'audio': [], 'video': [], 'pdf': []}
    st.session_state.play_logs = []
    st.session_state.current_day = 0
    st.session_state.playing_index = 0
    st.session_state.insights = []
    st.session_state.chat_history = []
    st.session_state.play_history = pd.DataFrame(columns=['day', 'type', 'file', 'time', 'sentiment'])

# ---------- FILE UPLOAD WITH PREVIEW ----------
st.sidebar.header("📤 Media Library")

def file_uploader_with_preview(label, types, key):
    files = st.sidebar.file_uploader(label, accept_multiple_files=True, type=types, key=key)
    
    if files:
        with st.sidebar.expander("📁 Uploaded Files Preview"):
            for file in files:
                if file not in st.session_state.media_files[key]:
                    st.session_state.media_files[key].append(file)
                    st.success(f"Added {file.name}")
                
                if key == 'pdf':
                    st.caption(f"📄 {file.name}")
                else:
                    st.caption(f"🎵 {file.name}")

file_uploader_with_preview("Upload Audio (MP3)", ['mp3'], 'audio')
file_uploader_with_preview("Upload Video (MP4)", ['mp4'], 'video')
file_uploader_with_preview("Upload PDFs", ['pdf'], 'pdf')

# Create combined playlist
all_files = []
for file in st.session_state.media_files['audio']:
    all_files.append(('audio', file))
for file in st.session_state.media_files['video']:
    all_files.append(('video', file))
for file in st.session_state.media_files['pdf']:
    all_files.append(('pdf', file))

# ---------- AI-POWERED PLAYBACK ----------
st.header("🎬 AI Media Experience")

def transcribe_audio(file_bytes):
    """Transcribe audio using OpenAI Whisper"""
    try:
        with st.spinner("🤖 AI is transcribing audio..."):
            result = ai_models['whisper'].transcribe(file_bytes)
            return result["text"]
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return ""

def generate_ai_insight(text):
    """Generate AI insight from text"""
    try:
        with st.spinner("🧠 Generating AI insights..."):
            summary = ai_models['summarizer'](text, max_length=150, min_length=30, do_sample=False)
            sentiment = ai_models['sentiment'](summary[0]['summary_text'])[0]
            return summary[0]['summary_text'], sentiment
    except Exception as e:
        st.error(f"Insight generation failed: {str(e)}")
        return "AI insight unavailable", {"label": "NEUTRAL", "score": 0}

def play_daily_file():
    if not all_files:
        return None
    
    # Get current file
    file_type, file = all_files[st.session_state.playing_index]
    
    # Process with AI
    ai_output = ""
    sentiment = {"label": "NEUTRAL", "score": 0}
    
    if file_type in ['audio', 'video']:
        # Save to temp file for processing
        with open("temp_media", "wb") as f:
            f.write(file.getbuffer())
        
        # Transcribe and analyze
        transcript = transcribe_audio("temp_media")
        ai_output, sentiment = generate_ai_insight(transcript)
        
        # Add to insights
        st.session_state.insights.append({
            'file': file.name,
            'transcript': transcript,
            'summary': ai_output,
            'sentiment': sentiment
        })
    
    # Add to logs
    log_entry = {
        'day': st.session_state.current_day + 1,
        'type': file_type,
        'file': file.name,
        'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'sentiment': sentiment['label']
    }
    st.session_state.play_logs.append(log_entry)
    
    # Update DataFrame for visualization
    st.session_state.play_history = st.session_state.play_history.append(log_entry, ignore_index=True)
    
    # Update indices
    st.session_state.playing_index = (st.session_state.playing_index + 1) % len(all_files)
    st.session_state.current_day += 1
    
    return file_type, file, ai_output, sentiment

# ---------- PLAYBACK UI ----------
if not all_files:
    st.warning("⚠️ Upload media files to begin your AI-powered media journey")
else:
    # Progress visualization
    progress_percent = (st.session_state.playing_index + 1) / len(all_files)
    st.subheader(f"📅 Day {st.session_state.current_day + 1} of {len(all_files)}")
    st.progress(progress_percent)
    
    # File type distribution
    file_types = [ft[0] for ft in all_files]
    type_counts = pd.Series(file_types).value_counts()
    fig = px.pie(type_counts, names=type_counts.index, values=type_counts.values, 
                 title="Media Distribution", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# Play button with animation
if st.button("▶️ Generate Today's AI Media Experience", use_container_width=True):
    if all_files:
        result = play_daily_file()
        if result:
            file_type, file, ai_output, sentiment = result
            
            # Display in elegant card
            with st.container():
                st.subheader(f"🎧 Now Playing: {file.name}")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display based on file type
                    if file_type == 'audio':
                        st.audio(file, format='audio/mp3')
                    elif file_type == 'video':
                        st.video(file, format='video/mp4')
                    elif file_type == 'pdf':
                        st.info("PDF scheduled for today - see summary below")
                    
                    # Sentiment indicator
                    sentiment_emoji = "😊" if sentiment['label'] == "POSITIVE" else "😐" if sentiment['label'] == "NEUTRAL" else "😞"
                    st.metric("Sentiment Analysis", f"{sentiment['label']} {sentiment_emoji}", 
                              f"{sentiment['score']*100:.1f}% confidence")
                
                with col2:
                    # AI output in styled card
                    if ai_output:
                        st.subheader("✨ AI Insights")
                        st.markdown(f"""
                        <div class="card">
                            <p style="font-size:16px; line-height:1.6">{ai_output}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("AI insights will appear here for audio/video files")
    else:
        st.error("Please upload media files first")

# ---------- ENHANCED PDF AI PROCESSING ----------
st.header("📚 AI-Powered Document Intelligence")

def pdf_to_text(file):
    """Extract text from PDF with formatting"""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages[:5]:  # First 5 pages
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return ""

def ai_pdf_qa(file, question):
    """Answer questions about PDF using AI"""
    text = pdf_to_text(file)
    if not text:
        return "Could not process PDF"
    
    try:
        result = ai_models['qa'](question=question, context=text)
        return result['answer']
    except Exception as e:
        return f"Error: {str(e)}"

if st.session_state.media_files['pdf']:
    # PDF selector
    pdf_names = [pdf.name for pdf in st.session_state.media_files['pdf']]
    selected_pdf = st.selectbox("Select a PDF document", pdf_names)
    
    if selected_pdf:
        pdf_file = [pdf for pdf in st.session_state.media_files['pdf'] if pdf.name == selected_pdf][0]
        
        # PDF summary
        with st.expander("📝 AI Summary"):
            text = pdf_to_text(pdf_file)
            if text:
                summary = ai_models['summarizer'](text, max_length=300, min_length=100)[0]['summary_text']
                st.markdown(f"**AI Summary of {selected_pdf}**")
                st.write(summary)
        
        # Interactive Q&A
        st.subheader("❓ Ask Questions About This Document")
        question = st.text_input("Enter your question about the document", key="pdf_question")
        
        if question and st.button("Get AI Answer"):
            answer = ai_pdf_qa(pdf_file, question)
            st.markdown(f"""
            <div class="card">
                <h4>AI Answer:</h4>
                <p style="font-size:16px; color:var(--primary)">{answer}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Upload PDF documents to enable AI-powered document analysis")

# ---------- MEDIA ANALYTICS DASHBOARD ----------
st.header("📊 AI-Powered Media Analytics")

if not st.session_state.play_history.empty:
    # Sentiment over time
    st.subheader("Sentiment Journey")
    sentiment_df = st.session_state.play_history.copy()
    sentiment_df['date'] = pd.to_datetime(sentiment_df['time'])
    
    fig = px.line(sentiment_df, x='date', y='day', color='sentiment',
                  title="Media Sentiment Over Time", markers=True,
                  color_discrete_map={
                      "POSITIVE": "green",
                      "NEUTRAL": "blue",
                      "NEGATIVE": "red"
                  })
    st.plotly_chart(fig, use_container_width=True)
    
    # Media type analytics
    st.subheader("Media Consumption Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        media_count = sentiment_df['type'].value_counts().reset_index()
        media_count.columns = ['Type', 'Count']
        fig = px.bar(media_count, x='Type', y='Count', color='Type',
                     title="Media Type Distribution", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(sentiment_df, names='sentiment', 
                     title="Overall Sentiment Distribution", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    # Full history
    st.subheader("Complete Play History")
    st.dataframe(sentiment_df.sort_values('day', ascending=False).reset_index(drop=True), 
                 use_container_width=True)
else:
    st.info("Play media files to unlock AI analytics")

# ---------- AI CHAT ASSISTANT ----------
st.header("💬 Media AI Assistant")

def chat_with_ai(query):
    """Generate response using media context"""
    context = f"""
    You are a media scheduling assistant. The user has scheduled {len(all_files)} media files. 
    Recent files: {[f[1].name for f in all_files[:3]] if all_files else 'None'}
    Today is day {st.session_state.current_day + 1} of the schedule.
    """
    
    try:
        response = f"AI Response to: {query} (This is a simulation. In a full implementation, this would connect to an LLM API)"
        return response
    except:
        return "I couldn't process your request. Please try again."

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your media schedule..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = chat_with_ai(prompt)
        st.markdown(response)
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# ---------- PERSONALIZATION & GAMIFICATION ----------
st.sidebar.header("🌟 Personalization")

# Achievement badges
st.sidebar.subheader("🏆 Achievements")
badges = {
    "First Play": st.session_state.current_day > 0,
    "Media Master": len(all_files) >= 5,
    "AI Explorer": any(st.session_state.insights),
    "Daily Champion": st.session_state.current_day >= 7
}

for badge, earned in badges.items():
    if earned:
        st.sidebar.success(f"✅ {badge}")
    else:
        st.sidebar.info(f"🔒 {badge}")

# Personalization options
st.sidebar.subheader("🎨 Customize Your Experience")
theme_color = st.sidebar.color_picker("Theme Color", "#6a11cb")
playback_speed = st.sidebar.slider("Playback Speed", 0.5, 2.0, 1.0, 0.25)

# ---------- FOOTER WITH AI TOUCH ----------
st.markdown("---")
st.caption(f"✨ AI-Powered Media Experience | Day {st.session_state.current_day + 1}")
st.caption(f"🧠 Powered by Whisper, BART, and Transformers | 👨‍🍳 Crafted by {st.session_state.username} for the 3MTT AI Showcase Challenge")
