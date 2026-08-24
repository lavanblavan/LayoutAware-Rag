"""
Serve the PaperRAG frontend on port 9010.
"""
import http.server
import os
import socketserver
import sys
import webbrowser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.session_log import configure_logging, get_logger

log = get_logger(__name__)

PORT = 9010
URL = f"http://localhost:{PORT}"
DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    configure_logging()
    os.chdir(DIR)
    log.info("PaperRAG frontend running at %s", URL)
    webbrowser.open(URL)
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()
