from pathlib import Path


def main() -> None:
    base = Path("data/raw")
    for part in ("policies", "projects", "faq", "chat_summaries"):
        (base / part).mkdir(parents=True, exist_ok=True)
    print("样例数据目录已就绪。")


if __name__ == "__main__":
    main()
