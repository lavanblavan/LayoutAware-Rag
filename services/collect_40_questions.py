"""Deprecated — use: python services/collect_questions.py --set train40"""
import sys

if __name__ == "__main__":
    if "--set" not in sys.argv and "--list" not in sys.argv and "--all" not in sys.argv:
        sys.argv = [sys.argv[0], "--set", "train40", *sys.argv[1:]]
    from services.collect_questions import main
    main()
