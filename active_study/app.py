"""Local read-only replay of saved active-study receipts.

This sidecar does not import the provider, discovery engine or final fields.
Only explicitly listed JSON receipts and rendered reports can be read.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from scripts.render_active_study import DEFAULT_OUTPUT, load_episode, load_results, safe_child

STATIC = Path(__file__).with_name('static') / 'index.html'
DOWNLOADS = {'REPORT.md': 'text/markdown; charset=utf-8', 'comparison.csv': 'text/csv; charset=utf-8',
             'summary.json': 'application/json; charset=utf-8', 'budget_curves.png': 'image/png'}


def make_server(directory=DEFAULT_OUTPUT, host='127.0.0.1', port=8767):
    directory = Path(directory).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_body(self, status, body, content_type='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path)
            try:
                if route.path == '/':
                    return self.send_body(200, STATIC.read_bytes(), 'text/html; charset=utf-8')
                if route.path == '/api/results':
                    comparison, summary, rows = load_results(directory)
                    return self.send_body(200, {'summary': summary, 'records': comparison['records'], 'rows': rows})
                if route.path == '/api/episode':
                    query = parse_qs(route.query)
                    if set(query) != {'task_id', 'policy'} or any(len(value) != 1 for value in query.values()):
                        return self.send_body(400, {'error': 'Provide exactly one task_id and one policy.'})
                    comparison, _, _ = load_results(directory)
                    episode = load_episode(directory, query['task_id'][0], query['policy'][0], comparison)
                    return self.send_body(200, episode)
                if route.path.startswith('/download/'):
                    name = route.path.removeprefix('/download/')
                    if name not in DOWNLOADS:
                        return self.send_body(404, {'error': 'Unknown report artifact.'})
                    return self.send_body(200, safe_child(directory, name).read_bytes(), DOWNLOADS[name])
                return self.send_body(404, {'error': 'Unknown read-only route.'})
            except FileNotFoundError:
                self.send_body(503, {'error': 'Saved results or episode are not available. Complete the run and render its report first.'})
            except KeyError:
                self.send_body(404, {'error': 'Requested saved result identity does not exist.'})
            except (ValueError, TypeError, IndexError) as exc:
                self.send_body(422, {'error': 'Saved receipt validation failed: '+str(exc)})

        def do_POST(self):
            self.send_body(405, {'error': 'Read-only replay: mutations and training are not available.'})

        do_PUT = do_PATCH = do_DELETE = do_POST

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    with make_server(args.output, args.host, args.port) as server:
        print(f'Read-only completed-experiment replay: http://{args.host}:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
