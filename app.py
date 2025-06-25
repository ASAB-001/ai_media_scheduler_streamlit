# AI Media Scheduler Web App (Streamlit version)

import streamlit as st
import os
import datetime
import random
import base64
import matplotlib.pyplot as plt
import pandas as pd
from PyPDF2 import PdfReader

# ---------- INIT ----------
st.set_page_config(page_title="AI Media Scheduler", layout="wide")
st.title("📅 AI Media Scheduler Dashboard")

if 'media_files' not in st.session_state:
    st.session_state.media_files = {'audio': [], 'video': [], 'pdf': []}
    st.session_state.play_logs = []
    st.session_state.current_day = 0

# ---------- FILE UPLOAD ----------
st.sidebar.header("📤 Upload Files")
audio_files = st.sidebar.file_uploader("Upload Audio Files (mp3)", accept_multiple_files=True, type=['mp3'])
video_files = st.sidebar.file_uploader("Upload Video Files (mp4)", accept_multiple_files=True, type=['mp4'])
pdf_files = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True, type=['pdf'])

for file in audio_files:
    st.session_state.media_files['audio'].append(file)
for file in video_files:
    st.session_state.media_files['video'].append(file)
for file in pdf_files:
    st.session_state.media_files['pdf'].append(file)

# ---------- DAILY PLAY ----------
def play_today_file():
    st.session_state.current_day += 1
    played_today = {}

    for filetype in ['audio', ' 'video', 'pdf']:
        files = st.session_state.media_files[filetype]
        if files:
            index = (st.session_state.current_day - 1) % len(files)
            selected = files[index]
            played_today[filetype] = selected.name

            # Add to logs
            st.session_state.play_logs.append({
                'day': st.session_state.current_day,
                'type': filetype,
                'file': selected.name,
                'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            })

    return played_today

# ---------- PDF SUMMARIZER ----------
def summarize_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages[:2]:
        text += page.extract_text() or ""
    return text[:500] + "..."

# ---------- PLAY BUTTON ----------
if st.button("▶️ Play Today’s Files"):
    result = play_today_file()
    st.success("Files scheduled today:")
    for typ, name in result.items():
        st.write(f"{typ.upper()}: {name}")

# ---------- PDF SUMMARY ----------
st.subheader("🧠 AI PDF Summarizer")
if st.session_state.media_files['pdf']:
    latest_pdf = st.session_state.media_files['pdf'][(st.session_state.current_day - 1) % len(st.session_state.media_files['pdf'])]
    st.info(f"Summarizing: {latest_pdf.name}")
    summary = summarize_pdf(latest_pdf)
    st.text_area("Summary:", summary, height=200)

# ---------- CHARTS ----------
st.subheader("📊 Progress Tracker")
if st.session_state.play_logs:
    df = pd.DataFrame(st.session_state.play_logs)
    chart = df.groupby(['day', 'type']).size().unstack().fillna(0)
    st.bar_chart(chart)
else:
    st.info("No progress yet. Click 'Play Today’s Files' to begin!")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("👨‍🍳 Crafted by Chef Boss for the 3MTT AI Showcase Challenge")
