from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import pandas as pd
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR, IdResolver, DidInMemoryCache

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

data = pd.DataFrame(columns=['text', 'created_at', 'author', 'uri', 'has_images', 'reply_to'])
collecting = False
stop_event = threading.Event()
data_lock = threading.Lock()

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
        with data_lock:
            global data
            data = pd.concat([data, pd.DataFrame([post_data])], ignore_index=True)

        print(f"Post adicionado: {post_data}")

    except Exception as e:
        print(f"Error processing post: {e}")

def collect_data():
    """Coleta dados por 10 segundos ou até ser interrompido."""
    global collecting
    collecting = True
    start_time = time.time()
    stop_event.clear()

    print("Iniciando coleta...")

    resolver = IdResolver(cache=DidInMemoryCache())
    client = FirehoseSubscribeReposClient()

    try:
        def process_message(message):
            if stop_event.is_set():
                client.stop()
                return

            print("Mensagem recebida:", message)

            commit = parse_subscribe_repos_message(message)
            if not hasattr(commit, 'ops'):
                return

            for op in commit.ops:
                if op.action == 'create' and op.path.startswith('app.bsky.feed.post/'):
                    car = CAR.from_bytes(commit.blocks)
                    for record in car.blocks.values():
                        if isinstance(record, dict) and record.get('$type') == 'app.bsky.feed.post':
                            process_post(record, commit.repo, op.path)

        client.start(process_message)

        while time.time() - start_time < 15 and not stop_event.is_set():
            time.sleep(0.1)

        stop_event.set()
        client.stop()
        collecting = False
        socketio.emit('collection_complete')
        print("Coleta concluída. Evento collection_complete emitido.")

    except Exception as e:
        print(f"Error during data collection: {e}")
        stop_event.set()
        collecting = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_collection():
    global collecting
    if not collecting:
        print("Iniciando a coleta via /start")
        threading.Thread(target=collect_data).start()
    return jsonify({'status': 'collecting'})

@app.route('/stop')
def stop_collection():
    global collecting
    stop_event.set()
    collecting = False
    print("Coleta interrompida via /stop")
    return jsonify({'status': 'stopped'})

@app.route('/data')
def get_data():
    with data_lock:
        print("Enviando dados coletados...")
        return jsonify(data.to_dict(orient='records'))

@app.route('/reset')
def reset_data():
    global data, collecting
    stop_event.set()
    with data_lock:
        data = pd.DataFrame(columns=['text', 'created_at', 'author', 'uri', 'has_images', 'reply_to'])
    collecting = False
    print("Dados resetados.")
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
