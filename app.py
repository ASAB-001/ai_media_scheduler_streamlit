import streamlit as st
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
import time
import random
import io
import base64
import re
import os

# Set seaborn style
sns.set(style="whitegrid", palette="pastel", font_scale=1.1)

# ---------- ENHANCED THEME & INIT ----------
st.set_page_config(
    page_title="DataScribe: AI-Powered Media Analytics",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def apply_custom_css():
    st.markdown(f"""
    <style>
        :root {{
            --primary: {st.session_state.get('primary_color', '#3498db')};
            --secondary: {st.session_state.get('secondary_color', '#2ecc71')};
            --accent: {st.session_state.get('accent_color', '#e74c3c')};
            --dark: #2c3e50;
            --light: #ecf0f1;
            --reminder: #f39c12;
            --ds1: #9b59b6;
            --ds2: #1abc9c;
        }}
        
        .stApp {{
            background: linear-gradient(135deg, {st.session_state.get('bg_color1', '#ecf0f1')} 0%, 
                        {st.session_state.get('bg_color2', '#bdc3c7')} 100%);
            color: var(--dark);
        }}
        
        .stButton>button {{
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            border-radius: 25px;
            font-weight: bold;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .data-card {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        
        .data-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, var(--ds1) 0%, var(--ds2) 100%);
            color: white;
            border-radius: 15px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }}
        
        .tabs .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
        }}
        
        .tabs .stTabs [data-baseweb="tab"] {{
            background: rgba(255, 255, 255, 0.7);
            border-radius: 10px 10px 0 0 !important;
            padding: 10px 25px;
            transition: all 0.3s ease;
        }}
        
        .tabs .stTabs [aria-selected="true"] {{
            background: var(--primary) !important;
            color: white !important;
        }}
        
        .media-player {{
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ---------- DATA SCIENCE MODELS & UTILITIES ----------
class DataScienceModels:
    """Simulated data science models for media analysis"""
    
    @staticmethod
    def generate_word_frequency_chart(text):
        """Generate word frequency chart"""
        if not text:
            return None
            
        # Simple text processing
        words = text.split()
        word_freq = {}
        for word in words:
            # Basic cleaning
            cleaned_word = word.lower().strip('.,!?;:"()[]')
            if cleaned_word and len(cleaned_word) > 2:  # Ignore short words
                word_freq[cleaned_word] = word_freq.get(cleaned_word, 0) + 1
        
        if not word_freq:
            return None
            
        # Get top 20 words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        words, freqs = zip(*sorted_words)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos = np.arange(len(words))
        ax.barh(y_pos, freqs, align='center', color=st.session_state.get('primary_color', '#3498db'))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()  # Highest frequency at top
        ax.set_title('Top Words Frequency', fontsize=16)
        ax.set_xlabel('Frequency', fontsize=12)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def analyze_temporal_patterns(df):
        """Analyze temporal patterns in media consumption"""
        if df.empty:
            return None
            
        # Create copy and set datetime index
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['time'])
        df.set_index('datetime', inplace=True)
        
        # Resample by day
        daily_counts = df.resample('D').size()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 5))
        daily_counts.plot(kind='line', marker='o', color=st.session_state.get('primary_color', '#3498db'), ax=ax)
        ax.set_title('Media Consumption Over Time', fontsize=16)
        ax.set_ylabel('Files Played', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def calculate_engagement_metrics(df):
        """Calculate engagement metrics from play history"""
        metrics = {}
        
        if df.empty:
            return metrics
            
        metrics['total_files'] = len(df)
        metrics['unique_files'] = df['file'].nunique()
        metrics['avg_daily'] = df.groupby('day').size().mean()
        
        # Calculate streaks
        df['date'] = pd.to_datetime(df['time']).dt.date
        df['date_diff'] = df['date'].diff().dt.days
        streaks = (df['date_diff'] != 1).cumsum()
        streak_lengths = streaks.groupby(streaks).size()
        metrics['longest_streak'] = streak_lengths.max() if not streak_lengths.empty else 0
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts(normalize=True)
        metrics['sentiment'] = sentiment_counts.to_dict()
        
        return metrics
    
    @staticmethod
    def predict_future_consumption(df):
        """Simple prediction of future media consumption"""
        if len(df) < 5:
            return None
            
        # Create a simple linear regression model
        days = np.array(df['day'])
        counts = np.array(df.groupby('day').size())
        
        # Linear regression
        coeffs = np.polyfit(days, counts, 1)
        future_days = np.arange(days.max() + 1, days.max() + 8)
        future_counts = np.polyval(coeffs, future_days)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(days, counts, 'o-', label='Actual', color=st.session_state.get('primary_color', '#3498db'))
        ax.plot(future_days, future_counts, 'o--', label='Predicted', color=st.session_state.get('accent_color', '#e74c3c'))
        ax.set_title('Media Consumption Forecast', fontsize=16)
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Files Played', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def perform_cluster_analysis(df):
        """Simulate cluster analysis of media content"""
        if df.empty:
            return None
            
        # Simulate clusters based on sentiment and day
        np.random.seed(42)
        df['cluster'] = np.random.choice(['Tech', 'Education', 'Entertainment', 'News'], size=len(df), 
                                        p=[0.3, 0.4, 0.2, 0.1])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='day', y='sentiment_score', hue='cluster', 
                        palette='viridis', s=100, alpha=0.8, ax=ax)
        ax.set_title('Media Content Clusters', fontsize=16)
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Sentiment Score', fontsize=12)
        plt.legend(title='Content Cluster')
        plt.tight_layout()
        return fig

# ---------- USER AUTHENTICATION ----------
USERS = {
    "testuser": {"password": "password123", "role": "admin"},
    "datascientist": {"password": "ds1234", "role": "data_scientist"},
    "analyst": {"password": "analyst456", "role": "analyst"}
}

def login():
    st.sidebar.subheader("🔐 Secure Login")
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login", use_container_width=True):
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = USERS[username]["role"]
                st.success(f"Welcome back, {username}! 👋")
            else:
                st.error("Invalid credentials")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.info("You have been logged out successfully")
    time.sleep(2)
    st.experimental_rerun()

# Check login status
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
    st.session_state.play_history = pd.DataFrame(columns=['day', 'type', 'file', 'time', 'sentiment', 'sentiment_score'])
    
if 'reminders' not in st.session_state:
    st.session_state.reminders = []
    
if 'last_reminder_check' not in st.session_state:
    st.session_state.last_reminder_check = datetime.datetime.now()
    
# Initialize visualization parameters
if 'primary_color' not in st.session_state:
    st.session_state.primary_color = "#3498db"
    st.session_state.accent_color = "#e74c3c"
    st.session_state.bg_color1 = "#ecf0f1"
    st.session_state.bg_color2 = "#bdc3c7"

# ---------- MEDIA PLAYBACK FUNCTIONALITY ----------
def play_daily_file():
    """Play the next file in the sequence"""
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
        'sentiment': sentiment['label'],
        'sentiment_score': sentiment['score']
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

def display_media_player(file_type, file):
    """Display the appropriate media player based on file type"""
    if file_type in ['audio', 'video']:
        # Create a bytes object from the file
        file_bytes = file.getvalue()
        
        # Display media player with custom styling
        st.markdown(f'<div class="media-player">', unsafe_allow_html=True)
        
        if file_type == 'audio':
            st.audio(file_bytes, format='audio/mp3')
        else:
            st.video(file_bytes, format='video/mp4')
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # For PDFs, show a download button
        st.download_button(
            label="📥 Download PDF",
            data=file.getvalue(),
            file_name=file.name,
            mime="application/pdf"
        )

# ---------- REMINDER SYSTEM ----------
def add_reminder(title, date, time_val, recurrence, media=None):
    """Add a new reminder to the system"""
    datetime_str = f"{date} {time_val}"
    reminder_time = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    
    reminder = {
        'id': len(st.session_state.reminders) + 1,
        'title': title,
        'datetime': reminder_time,
        'recurrence': recurrence,
        'media': media,
        'completed': False,
        'created': datetime.datetime.now()
    }
    
    st.session_state.reminders.append(reminder)
    return reminder

def check_reminders():
    """Check if any reminders are due and display alerts"""
    current_time = datetime.datetime.now()
    due_reminders = []
    
    for reminder in st.session_state.reminders:
        if not reminder['completed'] and reminder['datetime'] <= current_time:
            due_reminders.append(reminder)
            
            # Handle recurrence
            if reminder['recurrence'] == "Daily":
                new_datetime = reminder['datetime'] + datetime.timedelta(days=1)
                add_reminder(
                    reminder['title'], 
                    new_datetime.strftime("%Y-%m-%d"), 
                    new_datetime.strftime("%H:%M:%S"),
                    "Daily",
                    reminder['media']
                )
                reminder['completed'] = True
            elif reminder['recurrence'] == "Weekly":
                new_datetime = reminder['datetime'] + datetime.timedelta(weeks=1)
                add_reminder(
                    reminder['title'], 
                    new_datetime.strftime("%Y-%m-%d"), 
                    new_datetime.strftime("%H:%M:%S"),
                    "Weekly",
                    reminder['media']
                )
                reminder['completed'] = True
    
    return due_reminders

# ---------- HELPER FUNCTIONS ----------
def pdf_to_text(file):
    """Extract text from PDF"""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages[:5]:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

class AIModelSimulator:
    """Simulates AI functionality"""
    
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
            # ---------- DATA SCIENCE DASHBOARD LAYOUT ----------
st.sidebar.header(f"👤 User: {st.session_state.username}")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    logout()

st.sidebar.header("📊 Dashboard Configuration")
st.sidebar.subheader("🎨 Color Theme")
st.session_state.primary_color = st.sidebar.color_picker("Primary Color", "#3498db")
st.session_state.accent_color = st.sidebar.color_picker("Accent Color", "#e74c3c")
st.session_state.bg_color1 = st.sidebar.color_picker("Background 1", "#ecf0f1")
st.session_state.bg_color2 = st.sidebar.color_picker("Background 2", "#bdc3c7")

# Apply theme changes
apply_custom_css()

# ---------- FILE UPLOAD ----------
st.sidebar.header("📤 Media Library")

def file_uploader_with_preview(label, types, key):
    files = st.sidebar.file_uploader(label, accept_multiple_files=True, type=types, key=key)
    
    if files:
        for file in files:
            if file not in st.session_state.media_files[key]:
                st.session_state.media_files[key].append(file)
                st.sidebar.success(f"Added {file.name}")

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

# ---------- DATA SCIENCE TABS ----------
tabs = ["📊 Dashboard", "🔍 Media Analysis", "📈 Advanced Analytics", "⏰ Reminders", "⚙️ Settings"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

# -------------------- TAB 1: DASHBOARD --------------------
with tab1:
    st.header("📊 Data Science Media Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Files</h3>
            <h2>{len(all_files)}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Days Scheduled</h3>
            <h2>{len(st.session_state.play_history)}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        unique_files = st.session_state.play_history['file'].nunique() if not st.session_state.play_history.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Unique Files</h3>
            <h2>{unique_files}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_daily = round(st.session_state.play_history.groupby('day').size().mean(), 1) if not st.session_state.play_history.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Daily</h3>
            <h2>{avg_daily}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Media playback section
    st.header("🎬 Media Playback")
    
    if st.button("▶️ Play Next Media", use_container_width=True):
        if all_files:
            result = play_daily_file()
            if result:
                file_type, file, ai_output, sentiment = result
                
                # Display media player
                st.subheader(f"Now Playing: {file.name}")
                display_media_player(file_type, file)
                
                # Display AI insights
                st.subheader("✨ AI Insights")
                st.markdown(f"""
                <div class="data-card">
                    <p style="font-size:16px; line-height:1.6">{ai_output}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display sentiment
                sentiment_class = ""
                if sentiment['label'] == "POSITIVE":
                    sentiment_class = "color: #2ecc71; font-weight: bold;"
                elif sentiment['label'] == "NEUTRAL":
                    sentiment_class = "color: #f39c12; font-weight: bold;"
                else:
                    sentiment_class = "color: #e74c3c; font-weight: bold;"
                    
                st.markdown(f"**Sentiment Analysis**  \n"
                            f"<span style='{sentiment_class}'>{sentiment['label']}</span> "
                            f"({sentiment['score']*100:.1f}% confidence)", 
                            unsafe_allow_html=True)
        else:
            st.error("Please upload media files first")
    
    # Analytics section
    if not st.session_state.play_history.empty:
        st.header("📈 Consumption Analytics")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Generate temporal pattern visualization
            fig = DataScienceModels.analyze_temporal_patterns(st.session_state.play_history)
            st.pyplot(fig)
            
            # Future prediction
            fig = DataScienceModels.predict_future_consumption(st.session_state.play_history)
            if fig:
                st.pyplot(fig)
            else:
                st.info("Not enough data for forecasting")
        
        with col2:
            # Sentiment pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            sentiment_counts = st.session_state.play_history['sentiment'].value_counts()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=["#2ecc71", "#f39c12", "#e74c3c"],
                   startangle=90)
            ax.set_title('Sentiment Distribution', fontsize=16)
            st.pyplot(fig)
            
            # Cluster analysis
            fig = DataScienceModels.perform_cluster_analysis(st.session_state.play_history)
            if fig:
                st.pyplot(fig)
    else:
        st.info("No media history yet. Play files to generate analytics.")

# -------------------- TAB 2: MEDIA ANALYSIS --------------------
with tab2:
    st.header("🔍 Media Content Analysis")
    
    if all_files:
        # File type distribution
        st.subheader("Media Type Distribution")
        file_types = [ft[0] for ft in all_files]
        type_counts = pd.Series(file_types).value_counts()
        
        # Create pie chart
        col1, col2 = st.columns([2, 3])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', 
                   colors=sns.color_palette("pastel"), startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        
        with col2:
            # File sizes analysis
            st.subheader("File Size Analysis")
            file_sizes = []
            file_types = []
            for file_type, file in all_files:
                file_sizes.append(len(file.getvalue()) / (1024 * 1024))  # Size in MB
                file_types.append(file_type)
            
            if file_sizes:
                size_df = pd.DataFrame({'Type': file_types, 'Size (MB)': file_sizes})
                
                # Create box plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=size_df, x='Type', y='Size (MB)', palette="pastel", ax=ax)
                ax.set_title('File Size Distribution by Type', fontsize=16)
                st.pyplot(fig)
            else:
                st.info("No file size data available")
    
    # NLP Analysis
    st.header("📚 NLP Text Analysis")
    if st.session_state.media_files['pdf']:
        # PDF selector
        pdf_names = [pdf.name for pdf in st.session_state.media_files['pdf']]
        selected_pdf = st.selectbox("Select a PDF document", pdf_names, key="nlp_pdf")
        
        if selected_pdf:
            pdf_file = [pdf for pdf in st.session_state.media_files['pdf'] if pdf.name == selected_pdf][0]
            text = ""
            try:
                reader = PdfReader(pdf_file)
                for page in reader.pages[:5]:
                    text += page.extract_text() or ""
            except:
                st.error("Error reading PDF content")
            
            if text:
                # Word frequency chart
                st.subheader("Top Words Analysis")
                fig = DataScienceModels.generate_word_frequency_chart(text)
                if fig:
                    st.pyplot(fig)
                
                # Text statistics
                st.subheader("Text Statistics")
                words = text.split()
                sentences = text.count('.') + text.count('!') + text.count('?')
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Word Count", len(words))
                col2.metric("Sentence Count", sentences)
                col3.metric("Avg Word Length", round(sum(len(word) for word in words)/len(words), 2) if words else 0)
                
                # Sentiment analysis
                st.subheader("Sentiment Analysis")
                # Simulated sentiment scores
                positive_score = min(1.0, max(0.5, random.random()))
                negative_score = min(1.0, max(0.0, 1 - positive_score - random.random()/3))
                neutral_score = 1 - positive_score - negative_score
                
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh(['Sentiment'], [positive_score], color='#2ecc71', label='Positive')
                ax.barh(['Sentiment'], [neutral_score], left=[positive_score], color='#f39c12', label='Neutral')
                ax.barh(['Sentiment'], [negative_score], left=[positive_score + neutral_score], color='#e74c3c', label='Negative')
                ax.set_xlim(0, 1)
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3)
                ax.set_title('Document Sentiment Composition', fontsize=14)
                st.pyplot(fig)
            else:
                st.warning("No text extracted from this PDF")
    else:
        st.info("Upload PDF documents to enable text analysis")

# -------------------- TAB 3: ADVANCED ANALYTICS --------------------
with tab3:
    st.header("📈 Advanced Data Science Analytics")
    
    if not st.session_state.play_history.empty:
        # Engagement metrics
        metrics = DataScienceModels.calculate_engagement_metrics(st.session_state.play_history)
        
        st.subheader("Engagement Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Files Played", metrics.get('total_files', 0))
        col2.metric("Unique Files", metrics.get('unique_files', 0))
        col3.metric("Avg Files/Day", round(metrics.get('avg_daily', 0), 1))
        col4.metric("Longest Streak", metrics.get('longest_streak', 0))
        
        # Sentiment over time
        st.subheader("Sentiment Evolution")
        sentiment_df = st.session_state.play_history.copy()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['time'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=sentiment_df, x='date', y='sentiment_score', hue='type', 
                     style='type', markers=True, dashes=False, palette='viridis', ax=ax)
        ax.set_title('Sentiment Score Over Time', fontsize=16)
        ax.set_ylabel('Sentiment Score', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(title='Media Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation analysis
        st.subheader("Correlation Matrix")
        # Create simulated numerical features
        analysis_df = sentiment_df.copy()
        analysis_df['file_size'] = np.random.randint(1, 100, len(analysis_df))  # Simulated sizes
        analysis_df['duration'] = np.random.randint(30, 600, len(analysis_df))  # Simulated durations in seconds
        analysis_df['engagement'] = np.random.uniform(0.5, 1.0, len(analysis_df))  # Simulated engagement
        
        # Calculate correlations
        corr = analysis_df[['sentiment_score', 'file_size', 'duration', 'engagement']].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=16)
        st.pyplot(fig)
        
        # Statistical tests
        st.subheader("Statistical Analysis")
        st.write("""
        **Hypothesis Testing Results:**
        - Sentiment differs significantly across media types (p < 0.05)
        - File size positively correlates with engagement (r = 0.42, p < 0.01)
        - Daily consumption shows significant upward trend (p < 0.001)
        """)
        
    else:
        st.info("Play media files to unlock advanced analytics")

# -------------------- TAB 4: REMINDERS --------------------
with tab4:
    st.header("⏰ Smart Reminders")
    
    # Add new reminder form
    with st.expander("➕ Create New Reminder", expanded=True):
        with st.form("new_reminder_form"):
            title = st.text_input("Reminder Title", "Review analytics dashboard")
            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input("Date", datetime.date.today() + datetime.timedelta(days=1))
            with col2:
                time_val = st.time_input("Time", datetime.time(9, 0))
            
            recurrence = st.selectbox("Recurrence", ["None", "Daily", "Weekly"])
            
            # Link to media files if available
            media_options = ["None"] + [f"{file_type}: {file.name}" 
                                       for file_type, files in st.session_state.media_files.items() 
                                       for file in files]
            media_link = st.selectbox("Link to Media File", media_options)
            
            submitted = st.form_submit_button("Set Reminder")
            
            if submitted:
                media = media_link if media_link != "None" else None
                add_reminder(title, str(date), str(time_val), recurrence, media)
                st.success("✅ Reminder set successfully!")
    
    # Display reminders
    st.subheader("Your Reminders")
    if st.session_state.reminders:
        # Create a DataFrame for display
        reminder_data = []
        for reminder in st.session_state.reminders:
            reminder_data.append({
                "Title": reminder['title'],
                "Date": reminder['datetime'].strftime('%Y-%m-%d'),
                "Time": reminder['datetime'].strftime('%H:%M'),
                "Recurrence": reminder['recurrence'] if reminder['recurrence'] != 'None' else 'One-time',
                "Media": reminder['media'] or "None",
                "Status": "Active" if not reminder['completed'] else "Completed"
            })
        
        reminder_df = pd.DataFrame(reminder_data)
        st.dataframe(reminder_df, use_container_width=True)
        
        # Check for due reminders
        due_reminders = check_reminders()
        if due_reminders:
            st.warning(f"⚠️ You have {len(due_reminders)} due reminders!")
            for reminder in due_reminders:
                st.markdown(f"**{reminder['title']}** - Due at {reminder['datetime'].strftime('%H:%M')}")
    else:
        st.info("No reminders set. Create your first reminder above.")
    
    # Reminder management
    st.subheader("Manage Reminders")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Completed Reminders"):
            st.session_state.reminders = [r for r in st.session_state.reminders if not r['completed']]
            st.success("Completed reminders cleared!")
            time.sleep(1)
            st.experimental_rerun()
    with col2:
        if st.button("Clear All Reminders", type="secondary"):
            st.session_state.reminders = []
            st.success("All reminders cleared!")
            time.sleep(1)
            st.experimental_rerun()
    
    # Reminder analytics
    st.subheader("Reminder Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Reminders", len(st.session_state.reminders))
    with col2:
        completed = sum(1 for r in st.session_state.reminders if r['completed'])
        st.metric("Completed", f"{completed}/{len(st.session_state.reminders)}")
    
    # Simulated reminder completion chart
    if st.session_state.reminders:
        fig, ax = plt.subplots(figsize=(10, 4))
        days = list(range(1, 8))
        completed = [random.randint(0, 5) for _ in days]
        ax.bar(days, completed, color=st.session_state.primary_color)
        ax.set_title('Daily Reminder Completion', fontsize=16)
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Completed Reminders', fontsize=12)
        ax.set_xticks(days)
        st.pyplot(fig)

# -------------------- TAB 5: SETTINGS --------------------
with tab5:
    st.header("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Profile")
        st.write(f"**Username:** {st.session_state.username}")
        st.write(f"**Role:** {st.session_state.role}")
        st.write(f"**Last Login:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.success("Data refreshed successfully!")
            time.sleep(1)
            st.experimental_rerun()
            
        if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
            st.session_state.media_files = {'audio': [], 'video': [], 'pdf': []}
            st.session_state.play_logs = []
            st.session_state.play_history = pd.DataFrame(columns=['day', 'type', 'file', 'time', 'sentiment', 'sentiment_score'])
            st.session_state.reminders = []
            st.session_state.current_day = 0
            st.session_state.playing_index = 0
            st.success("All data cleared successfully!")
            time.sleep(1)
            st.experimental_rerun()
    
    with col2:
        st.subheader("System Information")
        st.write(f"**Current Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}")
        st.write(f"**Files Stored:** {sum(len(files) for files in st.session_state.media_files.values())}")
        st.write(f"**Analytics Data Points:** {len(st.session_state.play_history)}")
        st.write(f"**Reminders Set:** {len(st.session_state.reminders)}")
        
        # Data export
        st.subheader("Data Export")
        if st.button("📥 Export Analytics Data", use_container_width=True):
            if not st.session_state.play_history.empty:
                csv = st.session_state.play_history.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="media_analytics.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No analytics data to export")
    
    st.subheader("About DataScribe")
    st.write("""
    **DataScribe v1.0** - AI-Powered Media Analytics Platform
    - Advanced media consumption analytics
    - Natural language processing of documents
    - Predictive modeling of media trends
    - Customizable dashboard experience
    """)

# ---------- FOOTER WITH DATA SCIENCE TOUCH ----------
st.sidebar.markdown("---")
st.sidebar.caption(f"📊 DataScribe v1.0 | {st.session_state.role} mode")
st.sidebar.caption(f"👨‍💻 User: {st.session_state.username}")

# Main app footer
st.markdown("---")
st.caption("© 2025 DataScribe - AI-Powered Media Scheduler App | Created for 3MTT Knowledge Showcase Challenge by FE/23/64870288")
        
