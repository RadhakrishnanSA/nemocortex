#!/usr/bin/env python3
"""
NemoCortex Proxy Server
Handles CORS + forwards requests to NVIDIA API
Deploy on Render.com (free tier)
"""

import json
import time
import os
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import defaultdict

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
RATE_LIMIT = 20
RATE_WINDOW = 60

rate_store = defaultdict(list)

def check_rate_limit(ip):
    now = time.time()
    rate_store[ip] = [t for t in rate_store[ip] if now - t < RATE_WINDOW]
    if len(rate_store[ip]) >= RATE_LIMIT:
        return False
    rate_store[ip].append(now)
    return True

class ProxyHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

    def send_cors_headers(self):
        origin = self.headers.get('Origin', '')
        if 'github.io' in origin or origin.startswith('http://localhost') or origin.startswith('http://127') or origin == 'null':
            self.send_header('Access-Control-Allow-Origin', origin or '*')
        else:
            self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS, GET')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok', 'service': 'NemoCortex Proxy'}).encode())

    def do_POST(self):
        if self.path != '/api/chat':
            self.send_response(404)
            self.end_headers()
            return

        client_ip = self.client_address[0]
        if not check_rate_limit(client_ip):
            self.send_json_error(429, 'Rate limit exceeded. Max 20 requests/minute.')
            return

        length = int(self.headers.get('Content-Length', 0))
        if length > 50000:
            self.send_json_error(413, 'Request too large.')
            return

        try:
            body = self.rfile.read(length)
            payload = json.loads(body)
        except Exception as e:
            self.send_json_error(400, f'Invalid JSON: {str(e)}')
            return

        api_key = payload.get('api_key', '').strip()
        if not api_key or not api_key.startswith('nvapi-'):
            self.send_json_error(401, 'Invalid or missing API key.')
            return

        nvidia_payload = {k: v for k, v in payload.items() if k != 'api_key'}

        allowed_models = ['nvidia/nemotron-3-super-120b-a12b', 'nvidia/nemotron-3-nano-30b-a3b']
        if nvidia_payload.get('model') not in allowed_models:
            nvidia_payload['model'] = 'nvidia/nemotron-3-super-120b-a12b'

        nvidia_payload['max_tokens'] = min(int(nvidia_payload.get('max_tokens', 4096)), 8192)
        nvidia_payload['stream'] = False

        try:
            req_data = json.dumps(nvidia_payload).encode()
            req = urllib.request.Request(
                NVIDIA_API_URL,
                data=req_data,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                },
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                resp_body = resp.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(resp_body)

        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='replace')
            try:
                msg = json.loads(err_body).get('error', {}).get('message', err_body)
            except:
                msg = err_body[:200]
            self.send_json_error(e.code, msg)

        except urllib.error.URLError as e:
            self.send_json_error(502, f'Could not reach NVIDIA API: {str(e.reason)}')

        except Exception as e:
            self.send_json_error(500, f'Proxy error: {str(e)}')

    def send_json_error(self, code, message):
        body = json.dumps({'error': {'message': message}}).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), ProxyHandler)
    print(f'NemoCortex Proxy running on port {port}')
    server.serve_forever()
