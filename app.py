import streamlit as st
import time
import pandas as pd
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR, IdResolver, DidInMemoryCache

class BskyRealtimeAnalyzer:
    def __init__(self):
        self.initialize_session_state()
        self.last_post_placeholder = None

    def initialize_session_state(self):
        """Initialize the session state variables."""
        session_state_defaults = {
            'data': pd.DataFrame(columns=['text', 'created_at', 'author', 'uri', 'has_images', 'reply_to']),
            'collecting': False,
            'show_results': False,
            'pipeline_step': 0,  # 0: Coleta, 1: Tratamento, 2: Análise, 3: Finalizado
            'pipeline_status_text': "Aguardando início da coleta..."
        }
        for key, value in session_state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value


    def process_post(self, record, author, path):
        """Process a single post and insert it into the dataframe."""
        try:
            post_data = {
                'text': record.get('text', ''),
                'created_at': record.get('createdAt', ''),
                'author': author,
                'uri': f'at://{author}/{path}',
                'has_images': 'embed' in record,
                'reply_to': record.get('reply', {}).get('parent', {}).get('uri')
            }
            post_df = pd.DataFrame([post_data])
            st.session_state['data'] = pd.concat([st.session_state['data'], post_df], ignore_index=True)
            self.last_post_placeholder.text(f"Última postagem coletada: {post_data['text']}")
        except Exception as e:
            st.warning(f"Erro ao processar post: {e}")

    def process_message(self, message, client, start_time):
        """Process messages from the Firehose."""
        if time.time() - start_time >= 10:
            client.stop()
            st.session_state['collecting'] = False
            st.session_state['show_results'] = True
            st.session_state['pipeline_step'] = 1  # Coleta finalizada, próxima etapa: Tratamento
            st.session_state['pipeline_status_text'] = "Coleta de dados finalizada. Iniciando tratamento..."
            st.success("Captura finalizada!")
            st.rerun()
            return

        try:
            commit = parse_subscribe_repos_message(message)
            if not hasattr(commit, 'ops'):
                return

            for op in commit.ops:
                if op.action == 'create' and op.path.startswith('app.bsky.feed.post/'):
                    car = CAR.from_bytes(commit.blocks)
                    for record in car.blocks.values():
                        if isinstance(record, dict) and record.get('$type') == 'app.bsky.feed.post':
                            self.process_post(record, commit.repo, op.path)
        except Exception as e:
            st.warning(f"Erro ao processar mensagem: {e}")

    def collect_data(self):
        """Collect data for 10 seconds."""
        st.session_state['collecting'] = True
        st.session_state['pipeline_step'] = 0.5 # Iniciando coleta
        st.session_state['pipeline_status_text'] = "Iniciando coleta de dados..."
        start_time = time.time()
        resolver = IdResolver(cache=DidInMemoryCache())
        client = FirehoseSubscribeReposClient()

        with st.spinner("Aguarde, estamos capturando as postagens mais recentes..."):
            try:
                def message_handler(message):
                    self.process_message(message, client, start_time)

                client.start(message_handler)
                while st.session_state['collecting']:
                    time.sleep(0.1)
            except Exception as e:
                st.error(f"Erro durante a captura: {e}")
                st.session_state['collecting'] = False

    def perform_treatment(self):
        """Simula o tratamento dos dados."""
        st.session_state['pipeline_step'] = 1.5 # Iniciando tratamento
        st.session_state['pipeline_status_text'] = "Tratando os dados..."
        time.sleep(2) # Simulação do tratamento
        st.session_state['pipeline_step'] = 2 # Tratamento finalizado, próxima etapa: Análise
        st.session_state['pipeline_status_text'] = "Tratamento de dados finalizado. Iniciando análise..."
        st.rerun()

    def perform_analysis(self):
        """Simula a análise dos dados."""
        st.session_state['pipeline_step'] = 2.5 # Iniciando análise
        st.session_state['pipeline_status_text'] = "Analisando os dados..."
        time.sleep(3) # Simulação da análise
        st.session_state['pipeline_step'] = 3 # Análise finalizada
        st.session_state['pipeline_status_text'] = "Análise de dados concluída!"
        st.rerun()

    def render_ui(self):
        """Render the user interface."""
        st.title("Bsky Realtime Analyser & NLP Pipeline")

        # Barra de progresso
        progress_bar = st.progress(st.session_state['pipeline_step'] / 3.0)
        st.info(st.session_state['pipeline_status_text'])

        if not st.session_state['show_results']:
            if st.button("Iniciar captura") and not st.session_state['collecting']:
                st.session_state['data'] = pd.DataFrame(columns=['text', 'created_at', 'author', 'uri', 'has_images', 'reply_to'])
                st.session_state['show_results'] = False
                st.session_state['pipeline_step'] = 0
                st.session_state['pipeline_status_text'] = "Aguardando início da coleta..."
                self.last_post_placeholder = st.empty()
                self.collect_data()
            elif st.session_state['collecting']:
                st.info("Coletando dados... Aguarde.")
            else:
                st.info("Clique no botão para iniciar a coleta de dados.")
        else:
            if st.button("Refazer captura"):
                st.session_state['data'] = pd.DataFrame(columns=['text', 'created_at', 'author', 'uri', 'has_images', 'reply_to'])
                st.session_state['show_results'] = False
                st.session_state['pipeline_step'] = 0
                st.session_state['pipeline_status_text'] = "Aguardando início da coleta..."
                st.rerun()

            st.write("Postagens coletadas:")
            st.dataframe(st.session_state['data'])

            # Botões para as próximas etapas (simulação)
            if st.session_state['pipeline_step'] == 1:
                if st.button("Iniciar Tratamento"):
                    self.perform_treatment()
            elif st.session_state['pipeline_step'] == 2:
                if st.button("Iniciar Análise"):
                    self.perform_analysis()
            elif st.session_state['pipeline_step'] == 3:
                st.success("Pipeline concluído! Resultados da análise (simulados) abaixo.")
                # Aqui você exibiria os resultados da sua análise real

def main():
    analyzer = BskyRealtimeAnalyzer()
    analyzer.render_ui()

if __name__ == "__main__":
    main()