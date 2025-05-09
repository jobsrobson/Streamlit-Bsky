import streamlit as st
import subprocess
import os
import json
import time
import pandas as pd
from io import StringIO

LOG_FILE = "log.txt"

st.title("Bsky Realtime Analyser")

# Initialize session state
if 'process' not in st.session_state:
    st.session_state['process'] = None
if 'progress' not in st.session_state:
    st.session_state['progress'] = 0
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['text', 'created_at', 'author', 'uri', 'has_images', 'reply_to'])

# Function to clear log file
def clear_log_file():
    with open(LOG_FILE, "w") as log_file:
        log_file.write("")

# Function to run the scraper script
def run_scraper():
    clear_log_file()
    process = subprocess.Popen(['python', 'BskyScraper-All-60s.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process

# Function to read log file
def read_log_file():
    with open(LOG_FILE, "r") as log_file:
        return log_file.read()

# Function to process data in memory
def process_data(line):
    try:
        post_data = json.loads(line.strip())
        post_df = pd.DataFrame([post_data])
        st.session_state['data'] = pd.concat([st.session_state['data'], post_df], ignore_index=True)
    except json.JSONDecodeError:
        pass

# UI Components
if st.button("Iniciar captura"):
    st.session_state['process'] = run_scraper()
    st.session_state['progress'] = 0

progress_bar = st.progress(st.session_state['progress'])
log_container = st.empty()

# Real-time data processing
if st.session_state['process']:
    start_time = time.time()
    while st.session_state['process'].poll() is None:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / 60)
        st.session_state['progress'] = progress
        progress_bar.progress(progress)

        # Read output and process data
        output = st.session_state['process'].stdout.readline()
        if output:
            log_container.text(output.strip())
            process_data(output)

        time.sleep(1)

    st.session_state['process'] = None
    st.session_state['progress'] = 1.0
    progress_bar.progress(1.0)
    st.write("Captura finalizada!")

# Display collected data
if not st.session_state['data'].empty:
    st.write("Postagens coletadas:")
    st.dataframe(st.session_state['data'])
else:
    st.write("Nenhuma postagem coletada ainda.")
