"""Deprecated — use: python services/label_questions.py --set new30"""
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.session_log import configure_logging, get_logger
from services.label_questions import label_set

log = get_logger(__name__)

if __name__ == "__main__":
    configure_logging()
    log.info(json.dumps(label_set("new30"), indent=2))
