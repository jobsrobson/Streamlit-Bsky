import streamlit as st
import json
import time
import pandas as pd
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR, IdResolver, DidInMemoryCache

st.title("Bsky Realtime Analyser")

# Inicialização do estado da aplicação
if 'progress' not in st.session_state:
    st.session_state['progress'] = 0
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['text', 'created_at', 'author', 'uri', 'has_images', 'reply_to'])
if 'collecting' not in st.session_state:
    st.session_state['collecting'] = False

def process_post(record, author, path):
    """Processa um único post e o insere no dataframe."""
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
    except Exception as e:
        st.warning(f"Erro ao processar post: {e}")

def collect_data():
    """Coleta dados por 10 segundos."""
    st.session_state['collecting'] = True
    start_time = time.time()

    resolver = IdResolver(cache=DidInMemoryCache())
    client = FirehoseSubscribeReposClient()

    try:
        def process_message(message):
            if time.time() - start_time >= 10:
                client.stop()
                st.session_state['collecting'] = False
                st.success("Captura finalizada!")
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
                                process_post(record, commit.repo, op.path)
            except Exception as e:
                st.warning(f"Erro ao processar mensagem: {e}")

        # Inicia o cliente Firehose
        client.start(process_message)

        # Controle do progresso
        while st.session_state['collecting']:
            elapsed = time.time() - start_time
            st.session_state['progress'] = min(1.0, elapsed / 10)
            time.sleep(0.5)

    except Exception as e:
        st.error(f"Erro durante a captura: {e}")
        st.session_state['collecting'] = False

# Interface do usuário
if st.button("Iniciar captura") and not st.session_state['collecting']:
    # Limpa os dados anteriores
    st.session_state['data'] = pd.DataFrame(columns=['text', 'created_at', 'author', 'uri', 'has_images', 'reply_to'])
    st.session_state['progress'] = 0
    collect_data()

# Exibição da barra de progresso
st.progress(st.session_state['progress'])

# Exibição dos dados coletados
if not st.session_state['data'].empty:
    st.write("Postagens coletadas:")
    st.dataframe(st.session_state['data'])
else:
    st.write("Nenhuma postagem coletada ainda.")
# Exibição de mensagens de erro ou sucesso
if st.session_state['collecting']:
    st.info("Coletando dados... Aguarde.")
else:
    st.info("Clique no botão para iniciar a coleta de dados.")
