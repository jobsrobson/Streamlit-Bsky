import streamlit as st
import time
import json
import multiprocessing
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR, IdResolver, DidInMemoryCache

st.title('Coleta de Postagens no Bluesky')
st.write('Clique no botão para iniciar a coleta de postagens em tempo real do Bluesky por 60 segundos.')

processes = []
collected_data = multiprocessing.Manager().list()

def worker_process(queue, post_count, lock, stop_event, start_time, collected_data):
    resolver = IdResolver(cache=DidInMemoryCache())
    while not stop_event.is_set():
        try:
            message = queue.get(timeout=1)
            process_message(message, resolver, post_count, lock, start_time, collected_data)
        except multiprocessing.queues.Empty:
            continue
        except Exception as e:
            st.error(f"Worker error: {e}")


def client_process(queue, stop_event):
    client = FirehoseSubscribeReposClient()

    def message_handler(message):
        if stop_event.is_set():
            client.stop()
            return
        queue.put(message)

    try:
        client.start(message_handler)
    except Exception as e:
        st.error(f"Client process error: {e}")


def process_message(message, resolver, post_count, lock, start_time, collected_data):
    try:
        commit = parse_subscribe_repos_message(message)
        if not hasattr(commit, 'ops'):
            return

        for op in commit.ops:
            if op.action == 'create' and op.path.startswith('app.bsky.feed.post/'):
                author_handle = _resolve_author_handle(commit.repo, resolver)
                car = CAR.from_bytes(commit.blocks)
                for record in car.blocks.values():
                    if isinstance(record, dict) and record.get('$type') == 'app.bsky.feed.post':
                        post_data = _extract_post_data(record, author_handle)
                        collected_data.append(post_data)

    except Exception as e:
        st.error(f"Error processing message: {e}")


def _resolve_author_handle(repo, resolver):
    try:
        resolved_info = resolver.did.resolve(repo)
        return resolved_info.also_known_as[0].split('at://')[1] if resolved_info.also_known_as else repo
    except Exception as e:
        return repo


def _extract_post_data(record, author_handle):
    return {
        'text': record.get('text', ''),
        'created_at': record.get('createdAt', ''),
        'author': author_handle,
        'uri': record.get('uri', '')
    }

if st.button('Iniciar Coleta'):
    st.write('Iniciando a coleta...')
    post_count = multiprocessing.Value('i', 0)
    start_time = multiprocessing.Value('d', time.time())
    lock = multiprocessing.Lock()
    queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    collected_data = multiprocessing.Manager().list()

    client_proc = multiprocessing.Process(target=client_process, args=(queue, stop_event))
    client_proc.start()
    processes.append(client_proc)

    worker = multiprocessing.Process(target=worker_process, args=(queue, post_count, lock, stop_event, start_time, collected_data))
    worker.start()
    processes.append(worker)

    st.write("A coleta está em andamento. Aguarde 60 segundos ou clique em 'Encerrar Coleta'.")

if st.button('Encerrar Coleta'):
    st.write('Encerrando a coleta...')
    for proc in processes:
        proc.terminate()
        proc.join()
    st.success(f"Coleta finalizada! {len(collected_data)} postagens coletadas.")
    st.json(list(collected_data))

    # Exibição da visualização do JSON
    st.subheader("Visualização dos Dados Coletados")
    for post in collected_data:
        st.write(post)
