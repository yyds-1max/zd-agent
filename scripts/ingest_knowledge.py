import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipelines.ingest_pipeline import IngestPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="执行知识入库流程。")
    parser.add_argument("--source-dir", default="data/fixtures", help="知识源目录")
    args = parser.parse_args()

    result = IngestPipeline().run(args.source_dir)
    print(result)


if __name__ == "__main__":
    main()
