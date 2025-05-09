import streamlit as st
import subprocess
import os
import atproto
import json
import time
import argparse
import multiprocessing
import sys
import signal

# Diretório onde o script está localizado
SCRIPT_PATH = 'BskyScraper.py'

st.title('Coleta de Postagens no Bluesky')
st.write('Clique no botão para iniciar a coleta de postagens em tempo real do Bluesky por 60 segundos.')

if st.button('Iniciar Coleta'):
    st.write('Iniciando a coleta...')
    start_time = time.time()
    try:
        # Executa o script de coleta
        result = subprocess.run(['python', SCRIPT_PATH], capture_output=True, text=True)
        duration = time.time() - start_time
        st.write(f'Coleta finalizada em {duration:.2f} segundos.')
        st.text(result.stdout)
        if result.stderr:
            st.error(f'Erro durante a coleta: {result.stderr}')
    except Exception as e:
        st.error(f'Ocorreu um erro ao executar o script: {e}')

