import os
import sys
import time

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.session_log import configure_logging, get_logger

log = get_logger(__name__)

if __name__ == "__main__":
    configure_logging()
    start_time = time.perf_counter()
    url = "http://localhost:8000/ask"

    data = {
        "question": "What is the police arrest procedure?",
        "top_summary_docs": 3,
        "top_k_per_doc": 10,
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        log.info("%s", result)
    else:
        log.error("Error: %s %s", response.status_code, response.text)
    end_time = time.perf_counter()
    log.info("Request took %.2f seconds", end_time - start_time)
