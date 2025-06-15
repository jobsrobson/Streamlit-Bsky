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
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# Download das Stopwords do NLTK. Este bloco é executado uma vez no início da aplicação.
try:
  nltk.download('stopwords', quiet=True)
except Exception as e:
  # Não bloquear a UI se o download falhar. Um aviso é mostrado.
  st.toast(f"Alerta: Não foi possível baixar stopwords do NLTK: {e}. A modelagem de tópicos prosseguirá com as configurações padrão.", icon="⚠️")
  print(f"Alerta: Não foi possível baixar stopwords do NLTK: {e}.")


class BskyDataCollectorApp:
    """
    Classe principal que encapsula toda a lógica do aplicativo BskyMood.
    """

    def __init__(self):
        """
        Construtor da classe. Configura a página do Streamlit, inicializa
        variáveis e os estados da sessão.
        """
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
        self._initialize_topic_session_state()
        self.resolver_cache = DidInMemoryCache()
        self.sentiment_pipeline = None
        self.topic_model = None


    def _initialize_session_state(self):
        """
        Inicializa os estados da sessão do Streamlit para dados gerais e de coleta.
        Garante que as variáveis persistam entre as interações do usuário.
        """
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


    def _initialize_topic_session_state(self):
        """
        Inicializa os estados da sessão específicos para a análise de tópicos.
        """
        if 'topic_model_instance' not in st.session_state:
            st.session_state['topic_model_instance'] = None
        if 'topic_info_df' not in st.session_state:
            st.session_state['topic_info_df'] = pd.DataFrame()
        if 'topics_analyzed' not in st.session_state:
            st.session_state['topics_analyzed'] = False
        if 'performing_topic_analysis' not in st.session_state:
            st.session_state['performing_topic_analysis'] = False
        if 'texts_for_topic_analysis' not in st.session_state:
            st.session_state['texts_for_topic_analysis'] = []


    def _process_message(self, message, data_queue):
        """
        Processa uma única mensagem recebida do Firehose.
        Filtra por posts criados e os coloca na fila de dados.
        """
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


    def _lang_selector(self, text):
        """
        Detecta o idioma do texto e retorna True se for inglês, português ou espanhol.
        """
        try:
            lang = detect(text)
            return lang in ['en', 'pt', 'es']
        except Exception:
            return False


    def _extract_post_data(self, commit, op):
        """
        Extrai os dados relevantes de um post (skeet) a partir do objeto CAR.
        """
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


    def _collect_messages_threaded(self, stop_event, data_queue):
        """
        Inicia a coleta de mensagens em uma thread separada para não bloquear a UI.
        """
        client = FirehoseSubscribeReposClient()
        try:
            client.start(lambda message: self._process_message(message, data_queue))
        except Exception as e:
            st.toast(f"Erro na thread de coleta: {e}", icon=":material/dangerous:")
        finally:
            client.stop()


    def collect_data(self):
        """
        Gerencia o processo de coleta de dados, incluindo a UI (botão de parar, status).
        """
        stop_event = st.session_state['stop_event']
        data_queue = st.session_state['data_queue']
        st.session_state['collection_ended'] = False

        if st.session_state['collecting'] and not st.session_state['collection_ended']:
            stop_button_pressed = st.button("Parar Coleta", icon=":material/stop_circle:", help="Clique para parar a coleta de dados. Os dados já coletados serão mantidos na memória.")
            if stop_button_pressed:
                st.session_state['stop_event'].set()
                st.session_state['collecting'] = False

        start_time = time.time()
        collection_duration = st.session_state.get('collection_duration', 30)
        collecting_data_flag = st.session_state['collecting']

        if collecting_data_flag:
            collection_thread = threading.Thread(target=self._collect_messages_threaded, args=(stop_event, data_queue))
            collection_thread.daemon = True
            collection_thread.start()

            with st.status(f"Coletando posts do Bluesky durante {collection_duration} segundos. Aguarde!") as status:
                mensagens = [
                    "Estabelecendo conexão com o Firehose...", "Conexão estabelecida com sucesso!",
                    "Autenticando...", "Autenticação concluída!", "Organizando a fila...",
                    "Atualizando lista...", "Coletando posts... Isso pode demorar alguns minutos.",
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


    def preprocess_text(self, text):
        """
        Limpa e pré-processa o texto de uma publicação.
        """
        if not isinstance(text, str):
            return ''

        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}\b(/\S*)?', '', text, flags=re.IGNORECASE)
        text = emoji.demojize(text, language='en')
        return text

    def analyze_sentiment(self, status_obj):
        """
        Executa a análise de sentimentos nos dados coletados.
        """
        if not self.sentiment_pipeline:
            try:
                status_obj.update(label="Carregando modelo de análise de sentimentos...")
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
                    sentiment = "neutral" if not processed_text.strip() else self.sentiment_pipeline(processed_text)[0]['label']
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


    def perform_topic_modeling_and_sentiment(self, status_obj):
        """
        Executa a modelagem de tópicos e a análise de sentimentos agregada por tópico.
        """
        st.session_state['performing_topic_analysis'] = True
        st.session_state['topics_analyzed'] = False

        if not st.session_state.get('data'):
            st.warning("Não há dados coletados para a análise de tópicos.", icon="⚠️")
            status_obj.update(label="Nenhum dado para análise.", state="error", expanded=True)
            st.session_state['performing_topic_analysis'] = False
            return

        status_obj.update(label="Preparando textos para modelagem...")
        texts_for_bertopic = [self.preprocess_text(post.get('text', '')) for post in st.session_state['data']]
        st.session_state['texts_for_topic_analysis'] = texts_for_bertopic

        if not any(texts_for_bertopic):
            st.warning("Nenhum texto válido encontrado nos posts para a análise de tópicos.", icon="⚠️")
            status_obj.update(label="Nenhum texto para modelagem.", state="error", expanded=True)
            st.session_state['performing_topic_analysis'] = False
            return
        
        try:
            all_stop_words = list(nltk.corpus.stopwords.words('english')) + list(nltk.corpus.stopwords.words('portuguese')) + list(nltk.corpus.stopwords.words('spanish'))
            vectorizer_model = CountVectorizer(stop_words=all_stop_words)
        except Exception as e:
            st.warning(f"Não foi possível carregar stopwords: {e}. Usando BERTopic com configurações padrão.", icon="⚠️")
            vectorizer_model = None

        try:
            status_obj.update(label="Iniciando modelagem de tópicos com BERTopic... Isso pode levar alguns minutos.")
            self.topic_model = BERTopic(language="multilingual",
                                        vectorizer_model=vectorizer_model, 
                                        min_topic_size=3, 
                                        verbose=True)

            topics, _ = self.topic_model.fit_transform(texts_for_bertopic)
            st.session_state['topic_model_instance'] = self.topic_model

            status_obj.update(label="Modelagem concluída. Processando resultados...")

            if len(st.session_state['data']) == len(topics):
                for i, post_data in enumerate(st.session_state['data']):
                    post_data['topic_id'] = topics[i]
            
            topic_info_df = self.topic_model.get_topic_info()
            posts_df_for_topic_sentiment = pd.DataFrame(st.session_state['data'])

            if 'sentiment' in posts_df_for_topic_sentiment.columns and 'topic_id' in posts_df_for_topic_sentiment.columns:
                status_obj.update(label="Analisando sentimentos por tópico...")
                sentiment_by_topic = posts_df_for_topic_sentiment[posts_df_for_topic_sentiment['topic_id'] != -1] \
                                     .groupby('topic_id')['sentiment'].value_counts(normalize=True).unstack(fill_value=0)
                
                sentiment_by_topic = sentiment_by_topic.rename(columns=lambda x: f"{x.capitalize()} (%)" if x != 'analysis_error' else 'Error (%)')
                
                for col in sentiment_by_topic.columns:
                    sentiment_by_topic[col] = (sentiment_by_topic[col] * 100).round(1)

                if 'Topic' in topic_info_df.columns:
                    topic_info_df = topic_info_df.merge(sentiment_by_topic, left_on='Topic', right_index=True, how='left').fillna(0)
            
            st.session_state['topic_info_df'] = topic_info_df
            st.session_state['topics_analyzed'] = True
            status_obj.update(label="Análise de tópicos e sentimentos concluída!", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Erro durante a modelagem de tópicos: {e}", icon=":material/error:")
            status_obj.update(label=f"Erro na modelagem de tópicos: {e}", state="error", expanded=True)
        finally:
            st.session_state['performing_topic_analysis'] = False


    def display_data(self):
        """
        Renderiza a interface principal, exibindo dados, métricas, botões e resultados das análises.
        """
        if len(st.session_state['data']) > 0:
            df_collected = pd.DataFrame(st.session_state['data'])
            st.session_state['collected_df'] = df_collected
            num_rows = len(df_collected)

            num_has_images = df_collected['has_images'].sum() if 'has_images' in df_collected.columns else 0
            num_is_reply = df_collected['reply_to'].notna().sum() if 'reply_to' in df_collected.columns else 0

            if st.session_state['collection_ended'] and not st.session_state.get('performing_topic_analysis', False) and not st.session_state.get('collecting', False):
                if not st.session_state.get('topics_analyzed_toast_shown', False) and not st.session_state.get('sentiment_analysis_toast_shown', False):
                    st.toast(f"Ação finalizada com sucesso!", icon=":material/check_circle:")

            if 'sentiment' in df_collected.columns and st.session_state.get('sentiment_results') and not st.session_state.get('topics_analyzed'):
                total_analyzed = len(df_collected)
                sentiment_counts = df_collected['sentiment'].value_counts()
                positive_count = sentiment_counts.get('positive', 0)
                negative_count = sentiment_counts.get('negative', 0)
                neutral_count = sentiment_counts.get('neutral', 0)
                positive_percentage = (positive_count / total_analyzed) * 100 if total_analyzed > 0 else 0
                negative_percentage = (negative_count / total_analyzed) * 100 if total_analyzed > 0 else 0
                neutral_percentage = (neutral_count / total_analyzed) * 100 if total_analyzed > 0 else 0
                st.subheader("Resultados da Análise de Sentimentos")
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4, gap="small", border=True)
                with col_metric1: st.metric(label="Total de Posts Analisados", value=total_analyzed)
                with col_metric2: st.metric(label="Posts Positivos", value=f"{positive_percentage:.1f}%")
                with col_metric3: st.metric(label="Posts Negativos", value=f"{negative_percentage:.1f}%")
                with col_metric4: st.metric(label="Posts Neutros", value=f"{neutral_percentage:.1f}%")
            
            elif not st.session_state.get('topics_analyzed'):
                col1_metrics, col2_metrics, col3_metrics = st.columns(3, gap="small", border=True)
                with col1_metrics: st.metric(label="Total de Posts Coletados", value=num_rows)
                with col2_metrics: st.metric(label="Posts com Imagens", value=num_has_images)
                with col3_metrics: st.metric(label="Posts em Reply", value=num_is_reply)

            st.session_state['collected_df_for_download'] = df_collected

            if 'sentiment' in df_collected.columns and st.session_state.get('sentiment_results') and not st.session_state.get('topics_analyzed', False):
                
                st.sidebar.warning(
                    "- A análise de sentimentos é realizada automaticamente e pode não refletir a intenção original do autor.\n"
                    "- Menções e URLs são removidos durante a análise, mas são exibidos na tabela para registro."
                )
                columns_to_show = ['text', 'sentiment']
                st.dataframe(df_collected[columns_to_show], use_container_width=True)
            
            elif not df_collected.empty and not st.session_state.get('topics_analyzed', False):
                st.subheader("Dados Coletados")
                st.sidebar.warning(
                    "- Para executar a análise de sentimentos, clique em 'Analisar Sentimentos'.\n"
                    "- Para executar a análise de tópicos, conclua a análise de sentimentos primeiro.\n"
                    "- **Atenção**: as análises podem levar vários minutos, dependendo da sua conexão e do número de posts coletados."
                )
                cols_to_display = ['text', 'created_at', 'author', 'has_images', 'reply_to']
                st.dataframe(df_collected[cols_to_display], use_container_width=True)

            # Exibir botões de ação
            col1_buttons, col2_buttons, col3_buttons, col4_buttons = st.columns([1.7, 1.7, 1, 1])
            status_container_sentiment = st.empty()
            status_container_topics = st.empty()

            with col1_buttons:
                if not st.session_state.get('sentiment_results') and 'sentiment' not in df_collected.columns:
                    if st.button("Analisar Sentimentos", icon=":material/psychology:", use_container_width=True, type="primary", help="Clique para analisar os sentimentos dos posts coletados individualmente."):
                        with status_container_sentiment.status("Preparando para análise de sentimentos...", expanded=True) as status:
                            self.analyze_sentiment(status)
                        st.session_state['sentiment_analysis_toast_shown'] = True
                        st.rerun()
                elif 'sentiment' in df_collected.columns:
                    st.button("Analisar Sentimentos", icon=":material/psychology:", use_container_width=True, type="primary", help="Sentimentos individuais já analisados.", disabled=True)

            with col2_buttons:
                sentiment_analysis_done = 'sentiment' in df_collected.columns
                topics_already_analyzed = st.session_state.get('topics_analyzed', False)
                disable_topic_button = topics_already_analyzed or not sentiment_analysis_done

                if topics_already_analyzed: help_text = "Tópicos já analisados."
                elif not sentiment_analysis_done: help_text = "Execute a 'Análise de Sentimentos' primeiro."
                else: help_text = "Clique para extrair tópicos e analisar sentimentos por tópico."

                if st.button("Analisar Tópicos", icon=":material/hub:", use_container_width=True, type="primary", help=help_text, disabled=disable_topic_button):
                    with status_container_topics.status("Preparando para análise de tópicos...", expanded=True) as status_topic:
                        self.perform_topic_modeling_and_sentiment(status_topic)
                    st.session_state['topics_analyzed_toast_shown'] = True
                    st.rerun()

            with col3_buttons:
                if st.button("Reiniciar Coleta", on_click=lambda: st.session_state.update({
                    'data': [], 'collection_ended': False, 'collecting': False, 'sentiment_results': [], 
                    'collected_df': pd.DataFrame(), 'collected_df_for_download': pd.DataFrame(),
                    'stop_event': multiprocessing.Event(), 'data_queue': multiprocessing.Queue(),
                    'topic_model_instance': None, 'topic_info_df': pd.DataFrame(), 'topics_analyzed': False, 
                    'performing_topic_analysis': False, 'texts_for_topic_analysis': [],
                    'sentiment_analysis_toast_shown': False, 'topics_analyzed_toast_shown': False
                }), icon=":material/refresh:", help="Reinicie a coleta. Isso apagará todos os dados!", use_container_width=True):
                    pass

            with col4_buttons:
                df_to_download = pd.DataFrame(st.session_state['data']) if st.session_state['data'] else pd.DataFrame()
                if not df_to_download.empty:
                    st.download_button(
                        label="Baixar Dados", data=df_to_download.to_json(orient='records', indent=4, date_format='iso'),
                        file_name=f'bsky_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', mime='application/json',
                        help="Baixe os dados coletados (incluindo sentimentos e tópicos) em formato JSON.",
                        icon=":material/download:", use_container_width=True
                    )
                else:
                    st.button("Baixar Dados", disabled=True, use_container_width=True, help="Nenhum dado para baixar.", icon=":material/download:")
            
            if st.session_state.get('topics_analyzed', False) and not st.session_state.get('topic_info_df', pd.DataFrame()).empty:
                with st.container(border=True):
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Sentimentos por Tópicos", "🗺️ Mapa de Tópicos", "🗝️ Palavras-Chave", "🔍 Pesquisa de Tópicos", "📋 Dados Coletados"])

                    with tab1:
                        st.subheader("Análise de Sentimentos por Tópico")

                        st.sidebar.info(
                            "**Sobre a Análise de Tópicos:**\n\n"
                            "- Tópicos extraídos com BERTopic.\n"
                            "- Tópico '-1' agrupa posts considerados outliers.\n"
                            "- 'Palavras-Chave' são os termos mais significativos.\n"
                            "- Sentimento por Tópico é a distribuição percentual dos posts."
                        )

                        st.metric(label="Total de Tópicos Descobertos", value=len(st.session_state.get('topic_info_df', pd.DataFrame())))

                        source_topic_df = st.session_state['topic_info_df'].copy()
                        rename_map = {'Topic': 'ID Tópico', 'Count': 'Nº Posts', 'Name': 'Palavras-Chave'}
                        display_df = source_topic_df.rename(columns={k: v for k, v in rename_map.items() if k in source_topic_df.columns})

                        if 'Palavras-Chave' in display_df.columns and 'Name' in source_topic_df.columns:
                            display_df['Palavras-Chave'] = display_df['Palavras-Chave'].apply(lambda x: ", ".join(x.split('_')[1:]) if isinstance(x, str) and '_' in x else x)
                        
                        cols_for_main_display = [col for col in display_df.columns if col not in ['Representation', 'Representative_Docs', 'Representative_Samples']]
                        st.dataframe(display_df[cols_for_main_display], use_container_width=True)

                        topic_model_instance = st.session_state.get('topic_model_instance')
                        if topic_model_instance:
                            try:
                                num_topics_available = len(st.session_state['topic_info_df'])
                                if num_topics_available > 0:
                                    with tab2:
                                        st.subheader("Mapa de Distância Entre Tópicos")
                                        fig_topics = topic_model_instance.visualize_topics(top_n_topics=num_topics_available, title="")
                                        st.plotly_chart(fig_topics, use_container_width=True)

                                        with st.expander("🗺️ O que este Gráfico mostra?"):
                                            st.markdown("""
                                            - **Cada Círculo é um Tópico**: O tamanho indica a frequência (número de posts).
                                            - **Distância**: Círculos próximos representam tópicos semanticamente similares. Círculos distantes são sobre assuntos diferentes.
                                            - **Interatividade**: Clique em um círculo para ver seus tópicos mais relacionados.
                                            """)

                                    with tab3:
                                        st.subheader("Palavras Mais Importantes por Tópico")
                                        barchart_height = max(200, (num_topics_available * 3) + 0)
                                        fig_barchart = topic_model_instance.visualize_barchart(top_n_topics=num_topics_available, height=barchart_height, n_words=3, title="")
                                        st.plotly_chart(fig_barchart, use_container_width=True)
                                        
                                        with st.expander("📊 O que este Gráfico mostra?"):
                                            st.markdown("""
                                            - **Cada Sub-gráfico é um Tópico**: Detalha a composição de cada tópico individualmente.
                                            - **Comprimento das Barras**: Representa a importância de cada palavra para aquele tópico específico (score c-TF-IDF), não apenas sua frequência geral.
                                            """)
                                        topic_model_instance.visualize_hierarchy()

                            except Exception as e:
                                st.warning(f"Não foi possível gerar visualizações dos tópicos: {e}", icon="⚠️")
                with tab4:        
                    if st.session_state.get('topics_analyzed', False) and not st.session_state.get('topic_info_df', pd.DataFrame()).empty:
                        st.subheader("Pesquisar Tópico por Palavra-Chave")
                        search_term = st.text_input("Digite uma palavra-chave:", placeholder="Ex: economy, trump, brasil", help="Pesquise tópicos por palavras-chave. Exemplo: 'economy', 'trump', 'brasil'.")

                        if search_term:
                            if 'Palavras-Chave' in display_df.columns:
                                results_df = display_df[display_df['Palavras-Chave'].str.contains(search_term, case=False, na=False)]
                                if not results_df.empty:
                                    search_result_cols = ['ID Tópico', 'Palavras-Chave', 'Nº Posts', 'Positive (%)', 'Negative (%)', 'Neutral (%)']
                                    final_cols = [col for col in search_result_cols if col in results_df.columns]
                                    st.write(f"Resultados da busca para \"{search_term}\":")
                                    st.dataframe(results_df[final_cols], use_container_width=True)

                                    # Expander com os posts dos tópicos encontrados
                                    found_topic_ids = results_df['ID Tópico'].tolist()
                                    posts_in_found_topics = df_collected[df_collected['topic_id'].isin(found_topic_ids)]

                                    with st.expander(f"Ver posts dos tópicos encontrados na busca por '{search_term}'"):
                                        if not posts_in_found_topics.empty:
                                            posts_to_show = posts_in_found_topics[['text', 'sentiment', 'topic_id']]
                                            st.dataframe(posts_to_show, use_container_width=True)
                                        else:
                                            st.info("Não foram encontrados posts para os tópicos desta busca.")
                                else:
                                    st.info(f"Nenhum tópico encontrado com a palavra-chave \"{search_term}\".")

                    with tab5:
                        if st.session_state.get('topics_analyzed', False) and not df_collected.empty:
                            st.subheader("Dados Coletados Detalhados")
                            st.dataframe(df_collected, use_container_width=True)


        elif st.session_state['collection_ended'] and not st.session_state['data']:
            st.warning("Nenhum post foi coletado durante o período especificado ou que corresponda aos critérios.", icon="⚠️")
            if st.button("Tentar Nova Coleta", icon=":material/refresh:", use_container_width=True):
                self._reset_all_states()
                st.rerun()


    def _reset_all_states(self):
        """
        Função auxiliar para limpar todos os estados da sessão.
        """
        st.session_state.update({
            'data': [], 'collection_ended': False, 'collecting': False, 'sentiment_results': [], 
            'collected_df': pd.DataFrame(), 'collected_df_for_download': pd.DataFrame(),
            'stop_event': multiprocessing.Event(), 'data_queue': multiprocessing.Queue(),
            'topic_model_instance': None, 'topic_info_df': pd.DataFrame(), 'topics_analyzed': False, 
            'performing_topic_analysis': False, 'texts_for_topic_analysis': [],
            'sentiment_analysis_toast_shown': False, 'topics_analyzed_toast_shown': False
        })


    def run(self):
        """
        Método principal que executa o aplicativo.
        """
        st.markdown(f"<div style='text-align: left;'>{self.bskylogo_svg_template}</div>", unsafe_allow_html=True)
        st.text("")
        st.text("")

        st.sidebar.markdown(self.bskylogo_svg_template, unsafe_allow_html=True)
        st.sidebar.title("BskyMood")
        st.sidebar.markdown("**Coleta e Análise de Tópicos e Sentimentos no Bluesky**")

        if not st.session_state['collecting'] and not st.session_state['collection_ended']:
            st.warning("Nenhum post coletado ainda. Clique no botão 'Iniciar Coleta' para começar.", icon=":material/warning:")
            st.sidebar.info(
                "**Antes de começar**\n\n"
                "- Selecione um intervalo de coleta e clique em 'Iniciar Coleta'.\n"
                "- Intervalos maiores implicam em maior tempo de coleta e processamento.\n"
                "- As postagens coletadas podem incluir termos ofensivos ou inadequados."
            )

            st.session_state['collection_duration'] = st.sidebar.slider(
                "Duração da Coleta (segundos)", min_value=10, max_value=60, value=10, step=5,
                help="Defina por quanto tempo os posts serão coletados."
            )

            if st.sidebar.button("Iniciar Coleta", icon=":material/play_circle:", use_container_width=True, type="primary"):
                self._reset_all_states()
                st.session_state['collecting'] = True
                st.session_state['stop_event'].clear()
                st.rerun()

        elif st.session_state['collecting']:
            self.collect_data()

        if not st.session_state['collecting']:
            self.display_data()


if __name__ == "__main__":
    app = BskyDataCollectorApp()
    app.run()