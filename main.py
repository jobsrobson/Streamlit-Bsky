import streamlit as st
import pandas as pd
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR, IdResolver, DidInMemoryCache
from stqdm import stqdm
import time
import multiprocessing
import threading
import asyncio
from langdetect import detect
import queue
from datetime import datetime

class BskyDataCollectorApp:
    def __init__(self):
        st.set_page_config(page_title='BskyMood', layout='wide')
        self.svg_path_data = """
            M13.873 3.805C21.21 9.332 29.103 20.537 32 26.55v15.882c0-.338-.13.044-.41.867-1.512 4.456-7.418 21.847-20.923 7.944-7.111-7.32-3.819-14.64 9.125-16.85-7.405 1.264-15.73-.825-18.014-9.015C1.12 23.022 0 8.51 0 6.55 0-3.268 8.579-.182 13.873 3.805ZM50.127 3.805C42.79 9.332 34.897 20.537 32 26.55v15.882c0-.338.13.044.41.867 1.512 4.456 7.418 21.847 20.923 7.944 7.111-7.32 3.819-14.64-9.125-16.85 7.405 1.264 15.73-.825 18.014-9.015C62.88 23.022 64 8.51 64 6.55c0-9.818-8.578-6.732-13.873-2.745Z
        """
        self.bskylogo_svg_template = f"""
            <svg width="28" height="28" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
                <path d="{self.svg_path_data}" fill="#0085ff"/>
            </svg>
        """
        self._initialize_session_state()
        self.resolver_cache = DidInMemoryCache()

    def _initialize_session_state(self):
        if 'data' not in st.session_state:
            st.session_state['data'] = []
        if 'collecting' not in st.session_state:
            st.session_state['collecting'] = False
        if 'collection_ended' not in st.session_state:
            st.session_state['collection_ended'] = False
        if 'stop_event' not in st.session_state:
            st.session_state['stop_event'] = multiprocessing.Event()
        if 'start_time' not in st.session_state:
            st.session_state['start_time'] = 0.0
        if 'data_queue' not in st.session_state:
            st.session_state['data_queue'] = multiprocessing.Queue()

    def _process_message(self, message, data_queue):
        try:
            commit = parse_subscribe_repos_message(message)
            if not hasattr(commit, 'ops'):
                return

            for op in commit.ops:
                if op.action == 'create' and op.path.startswith('app.bsky.feed.post/'):
                    post_data = self._extract_post_data(commit, op)
                    if post_data and self._is_portuguese(post_data['text']):
                        data_queue.put(post_data)
        except Exception as e:
            print(f"Error processing message in thread: {e}")

    def _is_portuguese(self, text):
        try:
            return detect(text) == 'en'
        except Exception:
            return False

    def _extract_post_data(self, commit, op):
        try:
            car = CAR.from_bytes(commit.blocks)
            author_handle = commit.repo

            for record in car.blocks.values():
                if isinstance(record, dict) and record.get('$type') == 'app.bsky.feed.post':
                    return {
                        'text': record.get('text', ''),
                        'created_at': record.get('createdAt', ''),
                        'author': author_handle,
                        'uri': f'at://{commit.repo}/{op.path}',
                        'has_images': 'embed' in record,
                        'reply_to': record.get('reply', {}).get('parent', {}).get('uri')
                    }
        except Exception as e:
            print(f"Error extracting data: {e}")
            return None

    def _collect_messages_threaded(self, stop_event, data_queue):
        client = FirehoseSubscribeReposClient()
        try:
            client.start(lambda message: self._process_message(message, data_queue))
        except Exception as e:
            print(f"Erro no thread de coleta: {e}")
        finally:
            client.stop()

    def collect_data(self):
        stop_event = st.session_state['stop_event']
        data_queue = st.session_state['data_queue']
        st.session_state['collection_ended'] = False

        if st.session_state['collecting'] and not st.session_state['collection_ended']:
            stop_button_pressed = st.button("Parar Coleta", icon=":material/stop_circle:")
            if stop_button_pressed:
                st.session_state['stop_event'].set()
                st.session_state['collecting'] = False

        start_time = time.time()
        collection_duration = 5             # Duração da coleta em segundos	

        collecting_data = st.session_state['collecting']


        if collecting_data:
            collection_thread = threading.Thread(target=self._collect_messages_threaded, args=(stop_event, data_queue))
            collection_thread.daemon = True
            collection_thread.start()

            with st.spinner(f"Coletando posts do Bluesky durante {collection_duration} segundos. Aguarde!"):
                while collecting_data and not stop_event.is_set() and (time.time() - start_time < collection_duration):
                    try:
                        while not data_queue.empty():
                            st.session_state['data'].append(data_queue.get_nowait())
                    except queue.Empty:
                        pass
                    time.sleep(0.1)
                    collecting_data = st.session_state['collecting']

            st.session_state['collecting'] = False
            st.session_state['collection_ended'] = True
            stop_event.set()
            st.rerun()


    # Função de pré-processamento de texto
    def preprocess_text(self, text):
       
        return text
    
    # Função de análise de sentimentos
    def analyze_sentiment(self):
        st.session_state['sentiment_results'] = []
        if st.session_state['collection_ended'] and st.session_state['data']:
            for post in st.session_state['data']:
                processed_text = self.preprocess_text(post['text'])
                # Realize a análise de sentimento usando a biblioteca escolhida
                # Exemplo com NLTK:
                # scores = self.sentiment_analyzer.polarity_scores(processed_text)
                # sentiment = 'neutral'
                # if scores['compound'] >= 0.05:
                #     sentiment = 'positive'
                # elif scores['compound'] <= -0.05:
                #     sentiment = 'negative'
                # Exemplo com spaCy:
                # doc = self.nlp(processed_text)
                # sentiment = doc._.sentiment # Se usar um pipeline de sentimento do spaCy
                # Exemplo com Transformers:
                # result = self.sentiment_pipeline(processed_text)[0]
                # sentiment = result['label']

                # Por enquanto, vamos adicionar um sentimento placeholder
                sentiment = 'neutral'
                st.session_state['sentiment_results'].append({'uri': post['uri'], 'sentiment': sentiment})



    # Pós-Coleta - Exibe os dados coletados
    def display_data(self):
        if len(st.session_state['data']) > 0:
            num_rows = len(st.session_state['data'])
            st.success(f"Coleta finalizada com sucesso! {num_rows} posts foram coletados.", icon=":material/check_circle:")
            
            st.metric(label="Total de Posts Coletados", value=num_rows)

            df = pd.DataFrame(st.session_state['data'])
            st.write(df)
            st.session_state['collected_df'] = df

            left, middle, right = st.columns(3, vertical_alignment="bottom")
            with left:
                st.button("Próxima Etapa", icon=":material/arrow_forward:")  # Botão para ir para a próxima etapa
            with left:
                if st.button("Reiniciar Coleta", on_click=lambda: st.session_state.update({'data': [], 'collection_ended': False}), icon=":material/refresh:"):
                    self.collect_data()
                    st.rerun()
            with left:
                st.download_button(
                    label="Baixar Dados",
                    data=df.to_json(orient='records'),
                    file_name='bsky_data.json',
                    mime='application/json',
                    help="Baixe os dados coletados em formato JSON.",
                    icon=":material/download:"
                )
 
        else:
            st.error("Não há nenhum post na memória. Clique em Iniciar Coleta para coletar alguns!", icon=":material/error:")


    # Função Principal - Tela Inicial
    def run(self):
        st.markdown(self.bskylogo_svg_template, unsafe_allow_html=True)
        st.title("Bluesky Data Collector")

        st.sidebar.markdown(self.bskylogo_svg_template, unsafe_allow_html=True)
        st.sidebar.title("BskyMood")
        st.sidebar.text("Coleta e Análise de Sentimentos em Tempo Real no Bluesky")

        if not st.session_state['collecting'] and not st.session_state['collection_ended']:
            if st.sidebar.button("Iniciar Coleta", icon=":material/play_circle:"):
                st.session_state['collecting'] = True
                st.session_state['stop_event'].clear()
                st.session_state['data_queue'] = multiprocessing.Queue()
                self.collect_data()

        self.display_data()


if __name__ == "__main__":
    app = BskyDataCollectorApp()
    app.run()