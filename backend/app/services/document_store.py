import json
import os
from pathlib import Path
from ..models.schemas import DocumentInfo
from ..core.config import settings


class DocumentStore:
    """Persistent JSON registry for indexed research documents."""

    def __init__(self, storage_file: str | None = None):
        self.file_path = storage_file or str(Path(settings.BASE_DIR) / "data" / "documents.json")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.documents: dict[str, DocumentInfo] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = {k: DocumentInfo(**v) for k, v in data.items()}
            except Exception as e:
                print(f"Notice: Loading document store: {e}")
                self.documents = {}

    def _save(self) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({k: v.model_dump() for k, v in self.documents.items()}, f, indent=2)
        except Exception as e:
            print(f"Error saving document store: {e}")

    def add(self, doc_info: DocumentInfo) -> None:
        self.documents[doc_info.doc_id] = doc_info
        self._save()

    def get(self, doc_id: str) -> DocumentInfo | None:
        return self.documents.get(doc_id)

    def list_all(self) -> list[DocumentInfo]:
        return list(self.documents.values())

    def delete(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._save()
            return True
        return False


document_store = DocumentStore()
