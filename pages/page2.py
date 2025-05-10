import streamlit as st
import pandas as pd

st.title("Dados Coletados")

if 'collected_df' in st.session_state:
    df = st.session_state['collected_df']
    st.write("DataFrame Coletado:")
    st.write(df)
else:
    st.warning("Nenhum dado foi coletado ainda. Volte para a página principal e inicie a coleta.")

# Você pode adicionar um botão para voltar à página principal, se desejar
if st.button("Voltar para a Página Principal"):
    st.switch_page("main.py")