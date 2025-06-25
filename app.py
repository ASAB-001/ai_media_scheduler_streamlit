import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
import base64
from io import BytesIO
from PIL import Image
import textwrap
import random

# Set seaborn style for attractive visuals
sns.set(style="whitegrid", palette="pastel")

# --------- ENHANCED THEME & INIT ----------
st.set_page_config(
    page_title="AI Media Scheduler Pro",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)


# Custom theme and layout
st.set_page_config(
    page_title="AI Media Scheduler Pro",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #f0f0f0;
    font-family: Arial, sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #f7f9fc 0%, #e6f7ff 100%);
    color: #2d3436;
}
.stButton>button {
    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    color: #ffffff;
    border: none;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    cursor: pointer;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #2575fc 0%, #6a11cb 100%);
}
</style>
""", unsafe_allow_html=True)

# Set seaborn style
sns.set(style="whitegrid", palette="pastel")

# Your app code here
st.title("AI Media Scheduler Pro 🎬")
st.write("Welcome to the AI Media Scheduler Pro app!")

# Create columns
col1, col2 = st.columns(2)

# Add content to columns
col1.write("This is column 1")
col2.write("This is column 2")

# Add a plot
fig, ax = plt.subplots()
sns.set(style="whitegrid")
ax.plot([1, 2, 3, 4, 5])
st.pyplot(fig)


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
        
        .sentiment-positive {
            color: #2ecc71;
            font-weight: bold;
        }
        
        .sentiment-neutral {
            color: #f39c12;
            font-weight: bold;
        }
        
        .sentiment-negative {
            color: #e74c3c;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ---------- AI SIMULATION ----------
class AIModelSimulator:
    """Simulates AI functionality for environments without actual models"""
    
    @staticmethod
    def summarize_text(text):
        """Generate a simulated summary"""
        if not text:
            return "AI summary unavailable"
        
        sentences = text.split('. ')
        summary = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else text
        return summary
    
    @staticmethod
    def analyze_sentiment(text):
        """Simulate sentiment analysis"""
        if not text:
            return "NEUTRAL", 0.8
        
        positive_words = ["good", "great", "excellent", "positive", "happy", "success"]
        negative_words = ["bad", "poor", "negative", "unhappy", "failure", "problem"]
        
        positive_count = sum(word in text.lower() for word in positive_words)
        negative_count = sum(word in text.lower() for word in negative_words)
        
        if positive_count > negative_count:
            return "POSITIVE", min(0.9, 0.5 + positive_count/10)
        elif negative_count > positive_count:
            return "NEGATIVE", min(0.9, 0.5 + negative_count/10)
        else:
            return "NEUTRAL", 0.8
    
    @staticmethod
    def answer_question(context, question):
        """Simulate question answering"""
        if "what" in question.lower() and "date" in question.lower():
            return datetime.date.today().strftime("%B %d, %Y")
        elif "who" in question.lower():
            return "The author"
        elif "how" in question.lower() and "many" in question.lower():
            return "Several"
        else:
            answers = ["Based on the content, the answer is affirmative.", 
                      "The document suggests a negative outcome.",
                      "This requires further analysis.",
                      "The text indicates a positive result.",
                      "No definitive answer is provided."]
            return random.choice(answers)

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

def pdf_to_text(file):
    """Extract text from PDF"""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages[:5]:  # First 5 pages
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def play_daily_file():
    if not all_files:
        return None
    
    # Get current file
    file_type, file = all_files[st.session_state.playing_index]
    
    # Process with AI simulation
    ai_output = ""
    sentiment = {"label": "NEUTRAL", "score": 0.8}
    
    if file_type == 'pdf':
        text = pdf_to_text(file)
        ai_output = AIModelSimulator.summarize_text(text)
        sentiment["label"], sentiment["score"] = AIModelSimulator.analyze_sentiment(text)
    else:
        # Simulate AI processing for audio/video
        ai_output = f"AI analysis of {file.name}:\nThis content appears to be engaging and informative. Key themes include technology advancement and user experience optimization."
        sentiment["label"], sentiment["score"] = random.choice(
            [("POSITIVE", 0.85), ("NEUTRAL", 0.7), ("POSITIVE", 0.9)]
        )
    
    # Add to insights
    st.session_state.insights.append({
        'file': file.name,
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
    st.session_state.play_history = pd.concat([
        st.session_state.play_history, 
        pd.DataFrame([log_entry])
    ], ignore_index=True)
    
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
    
    # Create pie chart with matplotlib
    fig, ax = plt.subplots()
    ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', 
           colors=sns.color_palette("pastel"), startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie is circular
    st.pyplot(fig)

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
                    sentiment_class = ""
                    if sentiment['label'] == "POSITIVE":
                        sentiment_class = "sentiment-positive"
                    elif sentiment['label'] == "NEUTRAL":
                        sentiment_class = "sentiment-neutral"
                    else:
                        sentiment_class = "sentiment-negative"
                        
                    st.markdown(f"**Sentiment Analysis**  \n"
                                f"<span class='{sentiment_class}'>{sentiment['label']}</span> "
                                f"({sentiment['score']*100:.1f}% confidence)", 
                                unsafe_allow_html=True)
                
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

def ai_pdf_qa(file, question):
    """Answer questions about PDF using AI simulation"""
    text = pdf_to_text(file)
    return AIModelSimulator.answer_question(text, question)

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
                summary = AIModelSimulator.summarize_text(text)
                st.markdown(f"**AI Summary of {selected_pdf}**")
                st.markdown(f"<div class='card'>{summary}</div>", unsafe_allow_html=True)
        
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
    
    # Convert to datetime
    sentiment_df = st.session_state.play_history.copy()
    sentiment_df['date'] = pd.to_datetime(sentiment_df['time'])
    
    # Create line chart
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=sentiment_df, x='date', y='day', hue='sentiment', 
                 palette={"POSITIVE": "green", "NEUTRAL": "blue", "NEGATIVE": "red"},
                 marker="o", ax=ax)
    ax.set_title("Media Sentiment Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Day")
    st.pyplot(fig)
    
    # Media type analytics
    st.subheader("Media Consumption Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        st.markdown("**Media Type Distribution**")
        type_counts = sentiment_df['type'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=type_counts.index, y=type_counts.values, 
                    palette="pastel", ax=ax)
        ax.set_xlabel("Media Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    with col2:
        # Pie chart
        st.markdown("**Sentiment Distribution**")
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, 
               autopct='%1.1f%%', colors=["#2ecc71", "#f39c12", "#e74c3c"],
               startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
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
    # Simulate AI response
    responses = [
        f"Your media schedule currently has {len(all_files)} files scheduled. Next up: {all_files[st.session_state.playing_index][1].name if all_files else 'No files'}.",
        "Based on your viewing patterns, I recommend adding more video content for engagement.",
        "The sentiment analysis shows your audience responds well to positive content.",
        "You've scheduled content for the next {} days.".format(len(all_files) - st.session_state.playing_index),
        "Recent files have focused on technology and innovation themes."
    ]
    return random.choice(responses)

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
    "AI Explorer": len(st.session_state.insights) > 0,
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

# ---------- FOOTER WITH AI TOUCH ----------
st.markdown("---")
st.caption(f"✨ AI-Powered Media Experience | Day {st.session_state.current_day + 1}")
st.caption(f"🧠 Simulated AI Intelligence | 👨‍🍳 Crafted by {st.session_state.username} for the 3MTT AI Showcase Challenge")
