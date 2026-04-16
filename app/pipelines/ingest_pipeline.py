from pathlib import Path

from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.document_parser_service import DocumentParserService


class IngestPipeline:
    def __init__(self) -> None:
        self._repo = KnowledgeRepository()
        self._parser: DocumentParserService | None = None

    def run(self, source_dir: str) -> dict[str, str]:
        base = Path(source_dir)
        if not base.exists():
            return {"status": "error", "message": f"未找到数据目录：{source_dir}"}

        try:
            if self._parser is None:
                self._parser = DocumentParserService()
            chunks = self._parser.parse_directory(source_dir)
        except RuntimeError as exc:
            return {"status": "error", "message": str(exc)}
        if not chunks:
            return {"status": "error", "message": "未发现可入库的 txt 知识文档。"}

        try:
            saved_count = self._repo.save_chunks(chunks)
        except RuntimeError as exc:
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "message": f"入库完成，写入 {saved_count} 条知识片段。"}
