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
from transformers import pipeline

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
        self.sentiment_pipeline = None  # Inicialize o pipeline de análise de sentimentos

    # Função de Inicialização do estado da sessão do Streamlit
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
        if 'sentiment_results' not in st.session_state:
            st.session_state['sentiment_results'] = []
        if 'collected_df' not in st.session_state:
            st.session_state['collected_df'] = pd.DataFrame()

    # Função para Processamento de Mensagens Recebidas
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

    # Função para detecção de Idioma
    def _is_portuguese(self, text):
        try:
            return detect(text) == 'en'
        except Exception:
            return False

    # Função para extrair dados do Objeto CAR
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
            st.toast(f"Error extracting data: {e}", icon=":material/dangerous:")
            return None

    # Thread para coleta de mensagens em segundo plano
    def _collect_messages_threaded(self, stop_event, data_queue):
        client = FirehoseSubscribeReposClient()
        try:
            client.start(lambda message: self._process_message(message, data_queue))
        except Exception as e:
            st.toast(f"Error in collection thread: {e}", icon=":material/dangerous:")
        finally:
            client.stop()

    # Função de Coleta de Posts
    def collect_data(self):
        stop_event = st.session_state['stop_event']
        data_queue = st.session_state['data_queue']
        st.session_state['collection_ended'] = False

        if st.session_state['collecting'] and not st.session_state['collection_ended']:
            stop_button_pressed = st.button("Parar Coleta", icon=":material/stop_circle:", help="Clique para parar a coleta de dados. Os dados já coletados serão mantidos na memória.")
            if stop_button_pressed:
                st.session_state['stop_event'].set()
                st.session_state['collecting'] = False

        start_time = time.time()
        collection_duration = st.session_state.get('collection_duration', 10)  # Duração da coleta em segundos
        collecting_data = st.session_state['collecting']

        if collecting_data:
            collection_thread = threading.Thread(target=self._collect_messages_threaded, args=(stop_event, data_queue))
            collection_thread.daemon = True
            collection_thread.start()

            # Exibe mensagens de status enquanto coleta
            with st.status(f"Coletando posts do Bluesky durante {collection_duration} segundos. Aguarde!") as status:
                mensagens = [
                    "Estabelecendo conexão com o Firehose...",
                    "Conexão estabelecida com sucesso!",
                    "Autenticando...",
                    "Autenticação concluída!",
                    "Organizando a fila...",
                    "Atualizando lista...",
                    "Coletando posts... Isso pode demorar alguns minutos.",
                ]
                # Exibe cada mensagem uma vez, na ordem
                for i, msg in enumerate(mensagens):
                    status.update(label=msg)
                    # Durante as mensagens intermediárias, verifica se já deve parar
                    if i < len(mensagens) - 1:
                        time.sleep(1 if i != 0 else 2)
                        if stop_event.is_set() or (time.time() - start_time >= collection_duration):
                            break
                    else:
                        # Última mensagem: permanece até o fim da coleta
                        while collecting_data and not stop_event.is_set() and (time.time() - start_time < collection_duration):
                            try:
                                while not data_queue.empty():
                                    st.session_state['data'].append(data_queue.get_nowait())
                            except queue.Empty:
                                pass
                            time.sleep(0.5)
                            collecting_data = st.session_state['collecting']

            st.session_state['collecting'] = False
            st.session_state['collection_ended'] = True
            stop_event.set()
            st.rerun()


    # Função de pré-processamento de texto
    def preprocess_text(self, text):
        return text


    # Função de análise de sentimentos
    def analyze_sentiment(self, status_obj): # Aceita o objeto de status como argumento
        if not self.sentiment_pipeline:
            self.sentiment_pipeline = pipeline(
                model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                return_all_scores=False
            )


        st.session_state['sentiment_results'] = []
        updated_data = []
        if st.session_state['collection_ended'] and st.session_state['data']:
            for i, post in enumerate(st.session_state['data']):
                status_obj.update(label=f"Analisando post {i+1}/{len(st.session_state['data'])}: {post['text']}")
                try:
                    result = self.sentiment_pipeline(post['text'])[0]
                    sentiment = result['label']
                    updated_data.append({
                        'text': post['text'],
                        'created_at': post['created_at'],
                        'author': post['author'],
                        'uri': post['uri'],
                        'has_images': post['has_images'],
                        'reply_to': post['reply_to'],
                        'sentiment': sentiment
                    })
                    st.session_state['sentiment_results'].append({'text': post['text'], 'sentiment': sentiment})
                except Exception as e:
                    st.error(f"Erro ao analisar o sentimento do post {post['text']}: {e}", icon=":material/error:")
                    updated_data.append(post) # Mantém o post original em caso de erro
            status_obj.update(label="Análise de sentimentos concluída!", state="complete", expanded=False)
            st.session_state['data'] = updated_data
        else:
            st.error("Não há dados coletados para análise de sentimentos.", icon=":material/error:")
            return


    # Pós-Coleta - Exibe os dados coletados
    def display_data(self):
        if len(st.session_state['data']) > 0:
            df_collected = pd.DataFrame(st.session_state['data'])
            num_rows = len(df_collected)
            num_has_images = df_collected['has_images'].sum()
            num_is_reply = df_collected['reply_to'].notna().sum()

            st.toast(f"Coleta finalizada com sucesso!", icon=":material/check_circle:", )

            col1_metrics, col2_metrics, col3_metrics = st.columns(3, gap="small", border=True)
            with col1_metrics:
                st.metric(label="Total de Posts Coletados", value=num_rows)
            with col2_metrics:
                st.metric(label="Total de Posts com Imagens", value=num_has_images)
            with col3_metrics:
                st.metric(label="Total de Posts em Reply", value=num_is_reply)

            if not st.session_state.get('sentiment_results'):
                st.subheader("Dados Coletados")
                st.write(df_collected)
                st.session_state['collected_df'] = df_collected
            else:
                st.session_state['collected_df'] = pd.DataFrame() # Limpa o dataframe coletado do estado

            col1_buttons, col2_buttons, col3_buttons, col4_buttons = st.columns([1, 2, 1, 1], gap="small")
            status_container = st.empty() # Cria um container vazio para o status

            with col1_buttons:
                if st.button("Próxima Etapa", icon=":material/arrow_forward:", use_container_width=True, type="primary"):
                    with status_container.status("Preparando o ambiente para a análise de sentimentos...") as status:
                        self.analyze_sentiment(status) # Passa o objeto de status para a função
                    st.rerun()
            with col3_buttons:
                if st.button("Reiniciar Coleta", on_click=lambda: st.session_state.update({'data': [], 'collection_ended': False, 'sentiment_results': [], 'collected_df': pd.DataFrame()}), icon=":material/refresh:",
                                    help="Reinicie a coleta de dados. Isso apagará os dados em memória!", use_container_width=True):
                    self.collect_data()
                    st.rerun()
            with col4_buttons:
                st.download_button(
                    label="Baixar Dados",
                    data=df_collected.to_json(orient='records'),
                    file_name='bsky_data.json',
                    mime='application/json',
                    help="Baixe os dados coletados em formato JSON.",
                    icon=":material/download:",
                    use_container_width=True
                )

        else:
            st.error("Não há nenhum post na memória. Clique em Iniciar Coleta para coletar alguns!", icon=":material/error:")

        # Exibe os resultados da análise de sentimentos, se houver
        if st.session_state.get('sentiment_results'):
            st.subheader("Resultados da Análise de Sentimentos")
            sentiment_df = pd.DataFrame(st.session_state['sentiment_results'])
            st.write(sentiment_df)



    # Função Principal - Tela Inicial
    def run(self):
        st.markdown(self.bskylogo_svg_template, unsafe_allow_html=True)
        st.title("Bluesky Data Collector")

        st.sidebar.markdown(self.bskylogo_svg_template, unsafe_allow_html=True)
        st.sidebar.title("BskyMood")
        st.sidebar.text("Coleta e Análise de Sentimentos em Tempo Real no Bluesky")

        if not st.session_state['collecting'] and not st.session_state['collection_ended']:
            st.session_state['collection_duration'] = st.sidebar.slider(
                "Duração da Coleta (segundos)", min_value=10, max_value=120, value=10, step=1
            )
            if st.sidebar.button("Iniciar Coleta", icon=":material/play_circle:"):
                st.session_state['collecting'] = True
                st.session_state['stop_event'].clear()
                st.session_state['data_queue'] = multiprocessing.Queue()
                st.session_state['disable_start_button'] = True
                self.collect_data()

        self.display_data()

if __name__ == "__main__":
    app = BskyDataCollectorApp()
    app.run()