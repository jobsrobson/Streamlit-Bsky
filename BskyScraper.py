# Bluesky Firehose Scraper 0.1.0 
# Este script usa o cliente Firehose do Bluesky para coletar dados de postagens públicas e armazená-los em arquivos JSON.
# Esta versão coleta postagens em todos os idiomas.
# Veja o README.md para mais informações.

import subprocess
import sys
import os
# Verificar e instalar atproto se não estiver instalado
try:
    import atproto
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "atproto==0.0.61"])

from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR, IdResolver, DidInMemoryCache
import json
import time
import argparse
import multiprocessing
import sys
import signal

def worker_process(queue, output_file, verbose, post_count, lock, stop_event, start_time):
    resolver = IdResolver(cache=DidInMemoryCache())
    while not stop_event.is_set():
        try:
            message = queue.get(timeout=1)
            process_message(message, resolver, output_file, verbose, post_count, lock, start_time)
        except multiprocessing.queues.Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")

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

def process_message(message, resolver, output_file, verbose, post_count, lock, start_time):
    try:
        commit = parse_subscribe_repos_message(message)
        if not hasattr(commit, 'ops'):
            return

        for op in commit.ops:
            if op.action == 'create' and op.path.startswith('app.bsky.feed.post/'):
                _process_post(commit, op, resolver, output_file, verbose, post_count, lock, start_time)
    except Exception as e:
        print(f"Error processing message: {e}")

def _process_post(commit, op, resolver, output_file, verbose, post_count, lock, start_time):
    try:
        author_handle = _resolve_author_handle(commit.repo, resolver)
        car = CAR.from_bytes(commit.blocks)
        for record in car.blocks.values():
            if isinstance(record, dict) and record.get('$type') == 'app.bsky.feed.post':
                post_data = _extract_post_data(record, commit.repo, op.path, author_handle)
                _save_post_data(post_data, output_file, verbose, post_count, lock, start_time)
    except Exception as e:
        print(f"Error processing record: {e}")

def _resolve_author_handle(repo, resolver):
    try:
        resolved_info = resolver.did.resolve(repo)
        return resolved_info.also_known_as[0].split('at://')[1] if resolved_info.also_known_as else repo
    except Exception as e:
        print(f"Could not resolve handle for {repo}: {e}")
        return repo

def _extract_post_data(record, repo, path, author_handle):
    has_images = _check_for_images(record)
    reply_to = _get_reply_to(record)
    return {
        'text': record.get('text', ''),
        'created_at': record.get('createdAt', ''),
        'author': author_handle,
        'uri': f'at://{repo}/{path}',
        'has_images': has_images,
        'reply_to': reply_to
    }

def _check_for_images(record):
    embed = record.get('embed', {})
    return (
        embed.get('$type') == 'app.bsky.embed.images' or
        (embed.get('$type') == 'app.bsky.embed.external' and 'thumb' in embed)
    )

def _get_reply_to(record):
    reply_ref = record.get('reply', {})
    return reply_ref.get('parent', {}).get('uri')

def _save_post_data(post_data, output_file, verbose, post_count, lock, start_time):
    with lock:
        with open(output_file, 'a') as f:
            json.dump(post_data, f)
            f.write('\n')
    with post_count.get_lock():
        post_count.value += 1
        elapsed_time = time.time() - start_time.value
        sys.stdout.write(f"\rPosts coletados: {post_count.value} | Tempo decorrido: {elapsed_time:.2f} segundos")
        sys.stdout.flush()
    if verbose:
        print(f"\nSalvo post por @{post_data['author']}: {post_data['text'][:50]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect posts from the Bluesky firehose')
    parser.add_argument('-o', '--output', type=str, 
                        default=f"bluesky_posts_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
                        help='Output file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print each post as it is collected')
    args = parser.parse_args()

    post_count = multiprocessing.Value('i', 0)
    start_time = multiprocessing.Value('d', time.time())
    lock = multiprocessing.Lock()
    queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    resolver = IdResolver(cache=DidInMemoryCache())

    client_proc = multiprocessing.Process(target=client_process, args=(queue, stop_event))
    client_proc.start()
    workers = []
    for _ in range(4):
        p = multiprocessing.Process(target=worker_process, args=(queue, args.output, args.verbose, post_count, lock, stop_event, start_time))
        p.start()
        workers.append(p)
    
    try:
        while time.time() - start_time.value < 60:  # em segundos. 900=15 minutos 
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    finally:
        stop_event.set()
        client_proc.terminate()
        for p in workers:
            p.terminate()
        print("\nColeta finalizada!")
