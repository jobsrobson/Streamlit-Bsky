import streamlit as st
import pandas as pd
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR, IdResolver, DidInMemoryCache
import time
import multiprocessing
import threading
import regex as re
from langdetect import detect
import queue
from datetime import datetime
from transformers import pipeline
import emoji

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
        self.sentiment_pipeline = None  # Inicializa o pipeline de análise de sentimentos

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
                    if post_data and self._lang_selector(post_data['text']): 
                        data_queue.put(post_data)
        except Exception as e:
            print(f"Error processing message in thread: {e}")


    # Função para detecção de Idioma
    def _lang_selector(self, text):
        try:
            lang = detect(text)
            return lang in ['en', 'pt', 'es']  # Aceita posts em inglês, português ou espanhol
        except Exception:
            return False

    # Função para extrair dados do Objeto CAR do Firehose
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
            st.toast(f"Erro ao extrair dados: {e}", icon=":material/dangerous:")
            return None


    # Thread para coleta de mensagens em segundo plano
    def _collect_messages_threaded(self, stop_event, data_queue):
        client = FirehoseSubscribeReposClient()
        try:
            client.start(lambda message: self._process_message(message, data_queue))
        except Exception as e:
            st.toast(f"Erro na thread de coleta: {e}", icon=":material/dangerous:")
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
        collection_duration = st.session_state.get('collection_duration', 30)   # 30 segundos de coleta por fallback
        collecting_data_flag = st.session_state['collecting'] 

        if collecting_data_flag:
            collection_thread = threading.Thread(target=self._collect_messages_threaded, args=(stop_event, data_queue))
            collection_thread.daemon = True 
            collection_thread.start()

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
                for i, msg in enumerate(mensagens):
                    status.update(label=msg)
                    if i < len(mensagens) - 1:
                        time.sleep(1 if i != 0 else 2) 
                        if stop_event.is_set() or (time.time() - start_time >= collection_duration):
                            break
                    else:
                        while collecting_data_flag and not stop_event.is_set() and (time.time() - start_time < collection_duration):
                            try:
                                while not data_queue.empty():
                                    st.session_state['data'].append(data_queue.get_nowait())
                            except queue.Empty:
                                pass 
                            except Exception as e:
                                print(f"Erro ao recuperar da fila: {e}")
                            time.sleep(0.5)
                            collecting_data_flag = st.session_state['collecting']
                            if not collecting_data_flag: 
                                stop_event.set()
            try:
                while not data_queue.empty():
                    st.session_state['data'].append(data_queue.get_nowait())
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Erro ao recuperar dados restantes da fila: {e}")

            st.session_state['collecting'] = False
            st.session_state['collection_ended'] = True
            stop_event.set() 
            if collection_thread.is_alive():
                 collection_thread.join(timeout=2) 
            st.rerun()


    # Função de pré-processamento de texto
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return '' # Retorna string vazia se não for string

        # 1. Remover menções
        text = re.sub(r'@\S+', '', text)
        # 2. Remover URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}\b(/\S*)?', '', text, flags=re.IGNORECASE)
         # 3. Transformar emojis em texto (ex: "🐍" -> ":cobra:")
        text = emoji.demojize(text, language='en')

        return text


    # Função de análise de sentimentos usando o pipeline do Hugging Face
    # O "lxyuan/distilbert-base-multilingual-cased-sentiments-student" é o modelo de análise de sentimentos
    def analyze_sentiment(self, status_obj): 
        if not self.sentiment_pipeline:
            try:
                self.sentiment_pipeline = pipeline(
                    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                    return_all_scores=False, 
                )
            except Exception as e:
                st.error(f"Erro ao carregar o modelo de análise de sentimentos: {e}", icon=":material/error:")
                status_obj.update(label="Falha ao carregar modelo de análise.", state="error", expanded=True)
                return

        st.session_state['sentiment_results'] = [] 
        updated_data_with_sentiment = []
        if st.session_state['collection_ended'] and st.session_state['data']:
            total_posts = len(st.session_state['data'])
            for i, post in enumerate(st.session_state['data']):
                status_obj.update(label=f"Analisando post {i+1}/{total_posts}: \"{post['text'][:50]}...\"")
                try:
                    processed_text = self.preprocess_text(post['text'])
                    if not processed_text.strip(): 
                        sentiment = "neutral" 
                    else:
                        result = self.sentiment_pipeline(processed_text)[0]
                        sentiment = result['label']
                    
                    post_with_sentiment = post.copy() 
                    post_with_sentiment['sentiment'] = sentiment
                    updated_data_with_sentiment.append(post_with_sentiment)
                    st.session_state['sentiment_results'].append({'text': post['text'], 'sentiment': sentiment})

                except Exception as e:
                    st.error(f"Erro ao analisar o sentimento do post \"{post['text'][:50]}...\": {e}", icon=":material/error:")
                    post_with_error = post.copy()
                    post_with_error['sentiment'] = 'analysis_error' 
                    updated_data_with_sentiment.append(post_with_error)

            status_obj.update(label="Análise de sentimentos concluída!", state="complete", expanded=False)
            st.session_state['data'] = updated_data_with_sentiment 
        else:
            st.error("Não há dados coletados para análise de sentimentos.", icon=":material/error:")
            status_obj.update(label="Nenhum dado para analisar.", state="error", expanded=True)
            return

    # Pós-Coleta - Exibe os dados coletados
    def display_data(self):
        if len(st.session_state['data']) > 0:
            df_collected = pd.DataFrame(st.session_state['data'])
            num_rows = len(df_collected) # num_rows é o total de posts, seja antes ou depois da análise
            
            # Estas métricas são calculadas independentemente de a análise de sentimento ter sido feita ou não,
            # mas só são relevantes para o estado "antes da análise de sentimento".
            num_has_images = df_collected['has_images'].sum() if 'has_images' in df_collected.columns else 0
            num_is_reply = df_collected['reply_to'].notna().sum() if 'reply_to' in df_collected.columns else 0

            if st.session_state['collection_ended']:
                st.toast(f"Ação finalizada com sucesso!", icon=":material/check_circle:")

                # ----- INÍCIO DAS ALTERAÇÕES NA LÓGICA DE EXIBIÇÃO DAS MÉTRICAS -----
                if 'sentiment' in df_collected.columns and st.session_state.get('sentiment_results'):
                    # Se a análise de sentimentos foi feita e há resultados, exibe as novas métricas
                    total_analyzed = len(df_collected) # Todos os posts que passaram pela tentativa de análise
                    sentiment_counts = df_collected['sentiment'].value_counts()
                    
                    positive_count = sentiment_counts.get('positive', 0)
                    negative_count = sentiment_counts.get('negative', 0)
                    neutral_count = sentiment_counts.get('neutral', 0)
                    # analysis_error_count = sentiment_counts.get('analysis_error', 0) # DEPRECATED: Usado para mostrar erros

                    # Calcula porcentagens baseadas no total de posts analisados
                    positive_percentage = (positive_count / total_analyzed) * 100 if total_analyzed > 0 else 0
                    negative_percentage = (negative_count / total_analyzed) * 100 if total_analyzed > 0 else 0
                    neutral_percentage = (neutral_count / total_analyzed) * 100 if total_analyzed > 0 else 0

                    # Usando 4 colunas para as novas métricas de sentimento
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4, gap="small", border=True)
                    with col_metric1:
                        st.metric(label="Total de Posts Analisados", value=total_analyzed)
                    with col_metric2:
                        st.metric(label="Posts Positivos", value=f"{positive_percentage:.1f}%")
                    with col_metric3:
                        st.metric(label="Posts Negativos", value=f"{negative_percentage:.1f}%")
                    with col_metric4:
                        st.metric(label="Posts Neutros", value=f"{neutral_percentage:.1f}%")
                else:
                    # Se a análise de sentimentos não foi feita, exibe as métricas originais
                    col1_metrics, col2_metrics, col3_metrics = st.columns(3, gap="small", border=True)
                    with col1_metrics:
                        st.metric(label="Total de Posts Coletados", value=num_rows)
                    with col2_metrics:
                        st.metric(label="Posts com Imagens", value=num_has_images)
                    with col3_metrics:
                        st.metric(label="Posts em Reply", value=num_is_reply)
                # ----- FIM DAS ALTERAÇÕES NA LÓGICA DE EXIBIÇÃO DAS MÉTRICAS -----

            st.session_state['collected_df_for_download'] = df_collected 

            if 'sentiment' in df_collected.columns and st.session_state.get('sentiment_results'):
                st.subheader("Resultados da Análise de Sentimentos")
                st.sidebar.title("")
                st.sidebar.warning(
                    "- A análise ainda não é 100% precisa. Erros de classificação podem ocorrer em algumas postagens.\n"
                    "- A análise de sentimentos é realizada automaticamente e pode não refletir a intenção original do autor da postagem.\n"
                    "- Menções e URLs são removidos durante a análise, mas ainda são exibidos na tabela para fins de registro.\n"
                    "- As postagens podem incluir termos ofensivos ou inadequados, pois não há filtragem de conteúdo."
                )

                columns_to_show = ['text', 'sentiment']
                available_columns = [col for col in columns_to_show if col in df_collected.columns]
                if available_columns == columns_to_show: 
                    sentiment_display_df = df_collected[available_columns]
                    st.dataframe(sentiment_display_df, use_container_width=True)
                elif available_columns: 
                    st.warning(f"Exibindo colunas disponíveis: {', '.join(available_columns)}. Algumas colunas solicitadas ('text', 'sentiment') podem estar ausentes.")
                    sentiment_display_df = df_collected[available_columns]
                    st.dataframe(sentiment_display_df, use_container_width=True)
                else: 
                    st.error("Não foi possível exibir os resultados da análise de sentimentos como solicitado. Exibindo dados completos.")
                    st.dataframe(df_collected, use_container_width=True) 
            elif not df_collected.empty:
                st.subheader("Dados Coletados")
                st.sidebar.warning(
                    "- Para executar a análise de sentimentos, clique em 'Analisar Sentimentos'.\n"
                    "- Atenção: a análise pode levar vários minutos, dependendo da velocidade da sua conexão e da quantidade de dados coletados.\n"
                    "- As postagens podem incluir termos ofensivos ou inadequados, pois não há filtragem de conteúdo.\n",
                )
                st.dataframe(df_collected, use_container_width=True)
            
            col1_buttons, col2_buttons, col3_buttons, col4_buttons = st.columns([1.5, 0.5, 1, 1]) 
            status_container = st.empty()

            with col1_buttons:
                if not st.session_state.get('sentiment_results') and 'sentiment' not in df_collected.columns:
                    if st.button("Analisar Sentimentos", icon=":material/psychology:", use_container_width=True, type="primary", help="Clique para analisar os sentimentos dos posts coletados. Dependendo da velocidade da sua conexão, isso pode demorar alguns minutos."):
                        with status_container.status("Preparando o ambiente para a análise de sentimentos...", expanded=True) as status:
                            self.analyze_sentiment(status)
                        st.rerun()

            with col3_buttons:
                if st.button("Reiniciar Coleta", on_click=lambda: st.session_state.update({
                    'data': [],
                    'collection_ended': False,
                    'collecting': False, 
                    'sentiment_results': [],
                    'collected_df': pd.DataFrame(), 
                    'collected_df_for_download': pd.DataFrame(),
                    'stop_event': multiprocessing.Event(), 
                    'data_queue': multiprocessing.Queue() 
                }), icon=":material/refresh:",
                                help="Reinicie a coleta de dados. Isso apagará os dados em memória!", use_container_width=True):
                    pass 

            with col4_buttons:
                df_to_download = pd.DataFrame(st.session_state['data']) if st.session_state['data'] else pd.DataFrame()
                if not df_to_download.empty:
                    st.download_button(
                        label="Baixar Dados",
                        data=df_to_download.to_json(orient='records', indent=4), 
                        file_name=f'bsky_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 
                        mime='application/json',
                        help="Baixe os dados coletados (incluindo sentimentos, se analisados) em formato JSON.",
                        icon=":material/download:",
                        use_container_width=True
                    )
                else:
                    st.button("Baixar Dados", disabled=True, use_container_width=True, help="Nenhum dado para baixar.")

        elif st.session_state['collection_ended'] and not st.session_state['data']:
             st.warning("Nenhum post foi coletado durante o período especificado ou que corresponda aos critérios.", icon="⚠️")
             if st.button("Tentar Nova Coleta", icon=":material/refresh:", use_container_width=True):
                st.session_state.update({
                    'data': [],
                    'collection_ended': False,
                    'collecting': False,
                    'sentiment_results': [],
                    'collected_df': pd.DataFrame(),
                    'collected_df_for_download': pd.DataFrame(),
                    'stop_event': multiprocessing.Event(),
                    'data_queue': multiprocessing.Queue()
                })
                st.rerun()
        else:
            pass

    # Função Principal - Tela Inicial
    def run(self):
        st.markdown(f"<div style='text-align: left;'>{self.bskylogo_svg_template}</div>", unsafe_allow_html=True)
        st.text("")
        st.text("")
        # st.title("Bluesky Data Collector & Mood Analyzer")

        st.sidebar.markdown(self.bskylogo_svg_template, unsafe_allow_html=True)
        st.sidebar.title("BskyMood")
        st.sidebar.markdown("**Coleta e Análise de Sentimentos em Tempo Real no Bluesky**")

        if not st.session_state['collecting'] and not st.session_state['collection_ended']:
            st.warning(
                "Nenhum post coletado ainda. Clique no botão 'Iniciar Coleta' para começar.",
                icon=":material/warning:"
            )
            st.sidebar.info(
                "**Antes de começar**\n\n"
                "- Selecione um intervalo de coleta e clique em 'Iniciar Coleta'.\n"
                "- Intervalos maiores implicam em maior tempo de coleta e processamento. Defina um intervalo menor de coleta caso sua conexão à Internet seja lenta.\n"
                "- As postagens podem incluir termos ofensivos ou inadequados, pois não há filtragem de conteúdo."
            )
            
            st.session_state['collection_duration'] = st.sidebar.slider(
                "Duração da Coleta (segundos)", min_value=10, max_value=300, value=10, step=5, 
                help="Defina por quanto tempo os posts serão coletados. Quanto mais longo, mais tempo de processamento será necessário."
            )
            
            if st.sidebar.button("Iniciar Coleta", icon=":material/play_circle:", use_container_width=True, type="primary"):
                st.session_state['data'] = []
                st.session_state['sentiment_results'] = []
                st.session_state['stop_event'].clear() 
                st.session_state['data_queue'] = multiprocessing.Queue() 
                st.session_state['collecting'] = True
                st.rerun() 

        elif st.session_state['collecting']:
            self.collect_data()
            
        if not st.session_state['collecting']: 
            self.display_data()
            

if __name__ == "__main__":
    app = BskyDataCollectorApp()
    app.run()