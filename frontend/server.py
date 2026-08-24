"""
Serve the Quill chatbot frontend on port 9000.
"""
import http.server
import socketserver
import os

PORT = 9010
DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    os.chdir(DIR)
    print(f"Quill frontend running at http://localhost:{PORT}", flush=True)
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()
