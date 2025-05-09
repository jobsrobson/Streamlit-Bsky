import multiprocessing
import json
import time
import sys
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR, IdResolver, DidInMemoryCache

# Function to handle incoming messages and process data
def worker_process(queue, stop_event):
    resolver = IdResolver(cache=DidInMemoryCache())
    while not stop_event.is_set():
        try:
            message = queue.get(timeout=1)
            process_message(message, resolver)
        except multiprocessing.queues.Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")

# Function to connect to Firehose and receive messages
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
        if not stop_event.is_set():
            print(f"Client process error: {e}")

# Function to process and format data
def process_message(message, resolver):
    try:
        commit = parse_subscribe_repos_message(message)
        if not hasattr(commit, 'ops'):
            return

        for op in commit.ops:
            if op.action == 'create' and op.path.startswith('app.bsky.feed.post/'):
                post_data = extract_post_data(commit, op, resolver)
                if post_data:
                    print(json.dumps(post_data))
                    sys.stdout.flush()
    except Exception as e:
        print(f"Error processing message: {e}")

# Function to extract relevant post data
def extract_post_data(commit, op, resolver):
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

if __name__ == "__main__":
    queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    client_proc = multiprocessing.Process(target=client_process, args=(queue, stop_event))
    client_proc.start()

    workers = []
    for _ in range(2):
        worker = multiprocessing.Process(target=worker_process, args=(queue, stop_event))
        worker.start()
        workers.append(worker)

    start_time = time.time()

    try:
        while time.time() - start_time < 10:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        client_proc.terminate()
        for worker in workers:
            worker.terminate()
        print("Coleta finalizada.")
