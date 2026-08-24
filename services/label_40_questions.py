"""Deprecated — use: python services/label_questions.py --set train40"""
import sys

if __name__ == "__main__":
    if "--set" not in sys.argv and "--list" not in sys.argv:
        sys.argv = [sys.argv[0], "--set", "train40", *sys.argv[1:]]
    from services.label_questions import main
    main()
