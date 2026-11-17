#!/usr/bin/env python3
"""Re-run the November 15 README workloads on the final 300 W deployment.

Adapted from readme-bench-decode.py and readme-bench-prefill.py: exact decode
prompts, generation limits, two repetitions, concurrency order, and prefill
inventory repetitions are retained. Prefill gets a new leading run marker to
guarantee a cold prefix. This script never modifies hardware or serving settings.
The streamed aggregate is actual client-observed throughput on the running
deployment, including any admission, GPU-copy or scheduling pauses.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
import urllib.request

MAX = 2048
DONE = re.compile(r"glm53: done req-(\d+) ([^\r\n]+)")
FIELDS = re.compile(r"([\w/]+)=([^ ]+)")
LEGACY = {
    "readme-bench-decode.py": "a77ec63b5a69e0609ccf7e09fbda99ae4b75e6cb00c9ebfb80efad51ca8d6a07",
    "readme-bench-prefill.py": "b3a6c93a0c705583a7794487a828f6c138b97b43cfaa317b7333b2029d226425",
}

# Unchanged from the November 15 published README benchmark.
prose = [
    'A night-shift railway dispatcher slowly reconstructs a mistake from twenty years earlier as a fog-bound station comes back to life.',
    'A retired coastal cartographer returns to a shrinking island and follows an old channel with a grandchild during a spring tide.',
    'A conservator repairing a damaged illuminated manuscript finds a sequence of marginal sketches that traces a forgotten journey across the Alps.',
    'An apprentice glassmaker takes over a shuttered workshop in winter and tries to reproduce a colored lens remembered from childhood.',
]
code = [
    'An event-sourced warehouse inventory ledger with typed events, optimistic version checks, idempotency, snapshots, JSON serialization, queries, and unittest tests.',
    'A persistent job scheduler with SQLite, dependency graphs, cycle detection, priority, retry policies, leases, recovery, and unittest tests.',
    'An incremental document index with a tokenizer, BM25 ranking, phrase searches, a small query parser, persistence, a CLI, and unittest tests.',
    'An append-only double-entry bookkeeping ledger with Decimal arithmetic, CSV import validation, reconciliation, reporting, a CLI, and unittest tests.',
]
prompts = {
    'prose': ['Write an original literary short story in English, at least 3,200 words, on this premise: '+x+
              ' Begin directly with the scene. Use sustained prose, specific physical detail, dialogue, and an evolving plot. Continue through at least eight substantial scenes. Do not outline or summarize.' for x in prose],
    'code': ['Write a complete, runnable Python 3.12 implementation of the following project in one self-contained source file: '+x+
             ' Use only the standard library, include type annotations, all error paths, and at least 20 meaningful unittest cases. Write at least 400 lines of real implementation and tests. Begin with the actual Python source and do not use placeholders or a prose explanation.' for x in code],
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def shell(argv):
    return subprocess.check_output(argv, text=True).strip()


def body(prompt, maximum=MAX, stream=True):
    result = {'model': 'model', 'messages': [{'role': 'user', 'content': prompt}],
              'max_tokens': maximum, 'temperature': 0,
              'chat_template_kwargs': {'enable_thinking': False}}
    if stream:
        result.update(stream=True, stream_options={'include_usage': True})
    return result


class Benchmark:
    def __init__(self, args):
        self.a = args
        self.api = args.api.rstrip('/')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        require(not args.out.exists(), f'Refusing to overwrite {args.out}')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        self.container = json.loads(shell(['docker', 'inspect', args.container]))[0]
        require(self.container['State']['Running'], 'Serving container is not running')
        logs = next(m['Source'] for m in self.container['Mounts'] if m['Destination'] == '/rank-logs')
        self.log = Path(logs) / 'rank-0.log'
        self.result = {'schema_version': 1, 'status': 'running', 'mode': args.mode,
                       'started_utc': datetime.now(timezone.utc).isoformat(), 'run_id': args.run_id,
                       'legacy_sources_sha256': LEGACY, 'api': self.api, 'container': args.container,
                       'requests': {}, 'decode_batches': [], 'prefill_runs': [],
                       'method': {
                           'decode_repetitions': 2, 'decode_concurrencies': [1, 4],
                           'visible_output_tokens_per_request': MAX, 'temperature': 0, 'thinking': False,
                           'aggregate_stream_decode': 'sum(completion_tokens)/(latest_last_content-earliest_first_content)',
                           'aggregate_stream_scope': 'Actual client-observed content throughput; includes all scheduling, admission and GPU-copy pauses within the content interval',
                           'per_request_stream_decode': '(completion_tokens-1)/(last_content-first_content)',
                           'server_decode': 'Auxiliary per-request native logged tok/s; its t_end is frozen before that request\'s tail persistence; never sum these rates to derive aggregate throughput',
                           'prefill': 'actual_prompt_tokens/native_logged_prefill_seconds, cached_tokens=0',
                           'trailing_completion': 'HTTP/SSE completion minus last content; this is not a measurement of all persistence time because final content can itself be deferred',
                           'native_emit_policy': 'Final content and completion precede optional asynchronous cache persistence; record the deployed binary identity'}}
        self.save()

    def save(self):
        temporary = self.a.out.with_suffix(self.a.out.suffix + '.tmp')
        temporary.write_text(json.dumps(self.result, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(self.a.out)

    def http(self, path, payload=None, timeout=10):
        data = None if payload is None else canonical(payload).encode()
        req = urllib.request.Request(self.api + path, data, {'Content-Type': 'application/json'})
        with self.opener.open(req, timeout=timeout) as response:
            return json.load(response)

    def status(self):
        with self.opener.open(self.api + '/metrics', timeout=5) as response:
            text = response.read().decode()
        values = []
        for field in ('running', 'waiting'):
            found = re.search(r'^tensorfold:requests_' + field + r'(?:\{[^}]*\})?\s+(\d+)$', text, re.M)
            require(found is not None, f'Missing {field} metric')
            values.append(int(found.group(1)))
        return tuple(values)

    def idle(self):
        for _ in range(45):
            if self.status() == (0, 0):
                return
            time.sleep(1)
        raise RuntimeError('Service did not become idle')

    def power(self):
        text = shell(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,power.limit',
                      '--format=csv,noheader,nounits'])
        rows = [line.rsplit(',', 1) for line in text.splitlines()]
        require(len(rows) == 4 and all(float(row[1]) == 300 for row in rows), f'Expected four GPUs capped at 300 W: {text}')
        return text

    def ready(self):
        require('TF_GLM53_GRAPHS=0' in self.container['Config']['Env'], 'Expected native graph-off configuration')
        require(any(e.startswith('TF_GLM53_LMCACHE_PORT=') for e in self.container['Config']['Env']), 'Expected final LMCache deployment')
        model = self.http('/v1/models')
        require(any(m.get('id') == 'model' and m.get('max_model_len', m.get('context_length')) == 1000000
                    for m in model.get('data', [])), 'Expected 1M model capacity')
        for rank in range(4):
            text = self.log.with_name(f'rank-{rank}.log').read_text(errors='replace')
            require('1000000 tokens in one shared KV pool' in text and
                    re.search(r'0 (?:concurrent )?decode graphs captured', text), f'Invalid rank {rank} startup')
        self.result['models'] = model
        self.result['power_before'] = self.power()
        self.result['runtime'] = {k: self.container['HostConfig'].get(k) for k in ('CpusetCpus', 'CpusetMems', 'Memory', 'MemorySwap')}
        self.result['runtime']['image'] = self.container['Image']
        self.result['runtime']['command'] = self.container['Config']['Cmd']
        self.result['runtime']['environment'] = [e for e in self.container['Config']['Env']
                                                 if e.startswith(('TF_GLM53_', 'NCCL_'))]
        self.result['runtime']['rank0_log'] = str(self.log)
        self.save()

    def register(self, key, payload):
        tokens = self.http('/tokenize', payload, timeout=30).get('tokens')
        require(isinstance(tokens, list) and all(isinstance(t, int) for t in tokens), 'Tokenizer did not return IDs')
        item = {'body': payload, 'prompt_tokens': len(tokens), 'request_sha256': sha(payload),
                'prompt_token_ids': tokens, 'prompt_ids_sha256': sha(tokens)}
        self.result['requests'][key] = item
        self.save()
        return item

    def offset(self):
        stat = self.log.stat()
        return stat.st_size, stat.st_dev, stat.st_ino

    def done_logs(self, start, expected):
        deadline = time.monotonic() + 15
        while True:
            stat = self.log.stat()
            require(stat.st_size >= start[0] and (stat.st_dev, stat.st_ino) == start[1:], 'Log rotated during batch')
            with self.log.open('rb') as stream:
                stream.seek(start[0]); raw = stream.read().decode(errors='replace')
            records = []
            for match in DONE.finditer(raw):
                fields = dict(FIELDS.findall(match.group(2)))
                required = ('prompt', 'cached', 'tokens', 'tok/s', 'ttft', 'prefill', 'decode', 'graphs', 'eager', 'finish', 'accepted')
                require(all(k in fields for k in required), f'Incomplete done record: {match.group(0)}')
                row = {'request_number': int(match.group(1)), 'raw': match.group(0), 'finish': fields['finish']}
                row.update({k: int(fields[k]) for k in ('prompt', 'cached', 'tokens', 'graphs', 'eager')})
                row.update({k: float(fields[k].removesuffix('s')) for k in ('ttft', 'prefill', 'decode')})
                row['server_decode_tok_s'] = float(fields['tok/s'])
                row['accepted'], row['drafted'] = map(int, fields['accepted'].split('/'))
                records.append(row)
            require(len(records) <= expected, 'Unrelated requests contaminated batch logs')
            if len(records) == expected:
                require(all(r['graphs'] == 0 and r['eager'] > 0 for r in records), 'Request graph-off verification failed')
                return raw, records
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Missing done records: {raw[-6000:]}')
            time.sleep(.1)

    def stream(self, lane, fixture, gate):
        req = urllib.request.Request(self.api + '/v1/chat/completions', canonical(fixture['body']).encode(),
                                     {'Content-Type': 'application/json'})
        gate.wait(timeout=15)
        start = time.perf_counter()
        first = last = usage = finish = None
        pieces, events = [], []
        reasoning = []
        with self.opener.open(req, timeout=900) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    break
                data = json.loads(payload)
                now = time.perf_counter()
                if data.get('usage'):
                    usage = data['usage']
                for choice in data.get('choices', []):
                    delta = choice.get('delta', {})
                    content = delta.get('content')
                    if content:
                        first = now if first is None else first
                        last = now
                        pieces.append(content)
                        events.append(now)
                    if delta.get('reasoning_content'):
                        reasoning.append(delta['reasoning_content'])
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
        done = time.perf_counter()
        require(usage and first is not None and last is not None and last > first, 'Invalid stream timing/usage')
        require(not reasoning and usage.get('completion_tokens_details', {}).get('reasoning_tokens') == 0,
                'Expected thinking disabled and zero reasoning tokens')
        require(usage['completion_tokens'] == MAX and finish == 'length', 'Expected exactly 2048 visible output tokens')
        require(usage['prompt_tokens'] == fixture['prompt_tokens'], 'Prompt token count changed')
        text = ''.join(pieces)
        require(text.strip() and '\ufffd' not in text, 'Invalid generated content')
        return {'lane': lane, 'start': start, 'first': first, 'last': last, 'done': done,
                'usage': usage, 'finish': finish, 'content': text, 'content_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'events': events, 'request_sha256': fixture['request_sha256'],
                'ttft_s': first - start, 'stream_decode_s': last - first,
                'stream_decode_tok_s': (MAX - 1) / (last - first),
                'trailing_completion_s': done - last, 'request_total_s': done - start}

    def decode_batch(self, kind, concurrency, repetition):
        self.idle()
        self.power()
        fixtures = [self.result['requests'][f'{kind}_{i}'] for i in range(concurrency)]
        offset = self.offset()
        gate = threading.Barrier(concurrency)
        samples = []
        print(json.dumps({'phase': 'decode', 'kind': kind, 'concurrency': concurrency,
                          'repetition': repetition, 'state': 'started'}), flush=True)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(self.stream, i + 1, fixture, gate) for i, fixture in enumerate(fixtures)]
            while not all(f.done() for f in futures):
                samples.append((time.perf_counter(), *self.status()))
                wait(futures, timeout=.2)
            lanes = [f.result() for f in futures]
        raw, records = self.done_logs(offset, concurrency)
        require(Counter((l['usage']['prompt_tokens'], l['usage']['completion_tokens']) for l in lanes) ==
                Counter((r['prompt'], r['tokens']) for r in records), 'Logs do not match response token counts')
        for lane in lanes:
            matches = [r for r in records if (r['prompt'], r['tokens']) ==
                       (lane['usage']['prompt_tokens'], lane['usage']['completion_tokens'])]
            lane['server_record'] = matches[0] if len(matches) == 1 else None
            lane['server_record_matching'] = 'unique_token_counts' if len(matches) == 1 else 'batch_only_ambiguous'
        earliest = min(l['first'] for l in lanes)
        latest = max(l['last'] for l in lanes)
        elapsed = max(l['done'] for l in lanes) - min(l['start'] for l in lanes)
        total = sum(l['usage']['completion_tokens'] for l in lanes)
        row = {'kind': kind, 'concurrency': concurrency, 'repetition': repetition,
               'lanes': lanes, 'server_records': records, 'raw_logs': raw,
               'total_visible_tokens': total, 'stream_decode_span_s': latest - earliest,
               'aggregate_stream_decode_tok_s': total / (latest - earliest),
               'server_decode_tok_s': [r['server_decode_tok_s'] for r in records],
               'end_to_end_s': elapsed, 'aggregate_end_to_end_tok_s': total / elapsed,
               'full_overlap_s': max(0, min(l['last'] for l in lanes) - max(l['first'] for l in lanes)),
               'observed_running_max': max(s[1] for s in samples), 'metric_samples': len(samples),
               'full_concurrency_samples': sum(s[1] == concurrency for s in samples)}
        self.result['decode_batches'].append(row)
        self.save()
        require(row['observed_running_max'] >= concurrency, 'Requested concurrent activity was not observed')
        print(json.dumps({k: v for k, v in row.items() if k not in ('lanes', 'server_records', 'raw_logs')}), flush=True)

    def decode(self):
        for kind in ('prose', 'code'):
            for i, prompt in enumerate(prompts[kind]):
                self.register(f'{kind}_{i}', body(prompt))
        for repetition in (1, 2):
            for kind in ('prose', 'code'):
                for concurrency in (1, 4):
                    self.decode_batch(kind, concurrency, repetition)
        summary = []
        for kind in ('prose', 'code'):
            for concurrency in (1, 4):
                rows = [r for r in self.result['decode_batches'] if (r['kind'], r['concurrency']) == (kind, concurrency)]
                rates = [r['aggregate_stream_decode_tok_s'] for r in rows]
                server = [v for r in rows for v in r['server_decode_tok_s']]
                summary.append({'kind': kind, 'concurrency': concurrency,
                                'aggregate_stream_decode_runs': rates, 'aggregate_stream_decode_median': statistics.median(rates),
                                'server_per_request_decode_tok_s': server, 'server_per_request_decode_median': statistics.median(server),
                                'server_per_request_decode_range': [min(server), max(server)]})
        self.result['decode_summary'] = summary
        self.save()

    def prefill(self):
        for repetition in (1, 2):
            for label, reps in (('16k', 4000), ('32k', 8000), ('66k', 16500)):
                secret = f'WREN-{1469 + reps}'
                prompt = f'Run {repetition} benchmark {self.a.run_id}. Warehouse receipt {label}. The verified emergency inventory code is {secret}.\n'
                prompt += ' Stored inventory shelf.' * reps
                prompt += '\nWhat was the verified emergency inventory code in the receipt? Begin with just the code on its own line, then write a detailed 1,200-word story about the warehouse inventory audit.'
                name = f'cold-{label}-r{repetition}'
                fixture = self.register(name, body(prompt, 256, False))
                self.idle()
                self.power()
                offset = self.offset()
                print(json.dumps({'phase': 'prefill', 'case': name, 'tokens': fixture['prompt_tokens'], 'state': 'started'}), flush=True)
                start = time.perf_counter()
                answer = self.http('/v1/chat/completions', fixture['body'], timeout=900)
                elapsed = time.perf_counter() - start
                raw, records = self.done_logs(offset, 1)
                record = records[0]
                content = answer['choices'][0]['message'].get('content') or ''
                line = content.strip().splitlines()[0].strip() if content.strip() else ''
                require(record['prompt'] == fixture['prompt_tokens'] == answer['usage']['prompt_tokens'], 'Prefill token mismatch')
                require(record['cached'] == 0, 'Cold prefill hit a cached prefix; use a new --run-id')
                require(record['prefill'] > 0, 'No prefill duration')
                row = {'case': name, 'tier': label, 'repetition': repetition, 'expected': secret,
                       'correct': line == secret, 'first_line': line, 'server_record': record,
                       'raw_logs': raw, 'response': answer, 'request_sha256': fixture['request_sha256'],
                       'prompt_tokens': fixture['prompt_tokens'], 'prefill_tok_s': fixture['prompt_tokens'] / record['prefill'],
                       'ttft_s': record['ttft'], 'request_total_s': elapsed}
                self.result['prefill_runs'].append(row)
                self.save()
                require(row['correct'] and record['tokens'] >= 192 and record['finish'] != 'cancelled', 'Prefill needle/output check failed')
                require(answer['usage'].get('completion_tokens_details', {}).get('reasoning_tokens') == 0, 'Prefill thinking must be off')
                print(json.dumps({k: v for k, v in row.items() if k not in ('raw_logs', 'response', 'server_record')}), flush=True)
        self.result['prefill_summary'] = []
        for label in ('16k', '32k', '66k'):
            rows = [r for r in self.result['prefill_runs'] if r['tier'] == label]
            rates = [r['prefill_tok_s'] for r in rows]
            self.result['prefill_summary'].append({'tier': label, 'prompt_tokens': [r['prompt_tokens'] for r in rows],
                                                   'runs_tok_s': rates, 'median_tok_s': statistics.median(rates),
                                                   'range_tok_s': [min(rates), max(rates)], 'correct': sum(r['correct'] for r in rows)})
        self.save()

    def run(self):
        self.ready()
        if self.a.mode in ('decode', 'all'):
            self.decode()
        if self.a.mode in ('prefill', 'all'):
            self.prefill()
        self.result['power_after'] = self.power()
        require(self.result['power_before'] == self.result['power_after'], 'GPU identity or power setting changed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container', required=True)
    p.add_argument('--api', default='http://127.0.0.1:18992')
    p.add_argument('--mode', choices=('all', 'decode', 'prefill'), default='all')
    p.add_argument('--run-id', required=True, help='Unique leading prefill marker; use a fresh ID for each complete run')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', a.run_id) is not None, 'Use a short plain run ID')
    benchmark = Benchmark(a)
    try:
        benchmark.run()
        benchmark.result['status'] = 'passed'
    except Exception as error:
        benchmark.result['status'] = 'failed'
        benchmark.result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        benchmark.result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        benchmark.save()
    print(json.dumps({'status': 'passed', 'out': str(a.out)}), flush=True)


if __name__ == '__main__':
    main()
