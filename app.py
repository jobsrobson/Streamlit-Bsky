import streamlit as st
import subprocess
import os
import json
import time
from pathlib import Path

LOG_FILE = "log.txt"
DATA_FILE = "bluesky_data.jsonl"

st.title("Bsky Realtime Analyser")

# Function to clear log file
def clear_log_file():
    with open(LOG_FILE, "w") as log_file:
        log_file.write("")

# Function to run the scraper script
def run_scraper():
    clear_log_file()
    with open(LOG_FILE, "w") as log_file:
        process = subprocess.Popen(['python', 'BskyScraper-All-60s.py', '-o', DATA_FILE], stdout=log_file, stderr=log_file)
    return process

# Function to read log file
def read_log_file():
    with open(LOG_FILE, "r") as log_file:
        return log_file.read()

# Function to read data file
def read_data_file():
    data = []
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as data_file:
            data = [json.loads(line) for line in data_file.readlines()]
    return data

# UI Components
if 'process' not in st.session_state:
    st.session_state['process'] = None

if 'progress' not in st.session_state:
    st.session_state['progress'] = 0

if st.button("Iniciar captura"):
    st.session_state['process'] = run_scraper()
    st.session_state['progress'] = 0

progress_bar = st.progress(st.session_state['progress'])
log_container = st.empty()

if st.session_state['process']:
    start_time = time.time()
    while st.session_state['process'].poll() is None:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / 60)
        st.session_state['progress'] = progress
        progress_bar.progress(progress)
        log_container.text(read_log_file())
        time.sleep(1)

    st.session_state['process'] = None
    st.session_state['progress'] = 1.0
    progress_bar.progress(1.0)
    st.write("Captura finalizada!")

# Display collected data
posts = read_data_file()
if posts:
    st.write("Postagens coletadas:")
    st.json(posts)
else:
    st.write("Nenhuma postagem coletada ainda.")
