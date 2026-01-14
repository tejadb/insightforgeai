"""
Document parsing utilities (local-only, no Supabase/path resolution).

This module is intentionally small and "standard unstructured usage":
- Input: local file path OR bytes (+ optional filetype/filename)
- Output: list of Unstructured "elements" (Title, NarrativeText, Table, etc.)

Later we can add:
- Supabase download + temp-file handling
- chunking for RAG
- element normalization and filtering (headers/footers)
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

from unstructured.partition.auto import partition
from unstructured.documents.elements import Element


@dataclass(frozen=True)
class ParseOptions:
    """
    Minimal knobs that matter for high-quality parsing.

    Notes:
    - For PDFs, unstructured supports strategies like "hi_res" (layout-aware).
    - OCR behavior is controlled by the strategy + local system deps (tesseract/poppler).
    """

    strategy: Optional[str] = "hi_res"
    infer_table_structure: bool = True
    languages: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Default needs to be created at runtime to avoid a shared mutable list.
        if self.languages is None:
            object.__setattr__(self, "languages", ["eng"])


class UnstructuredDocParser:
    """
    Thin wrapper around `unstructured.partition.auto.partition`.
    """

    def __init__(self, options: Optional[ParseOptions] = None) -> None:
        self.options = options or ParseOptions()

    def parse_path(self, file_path: str | Path) -> list[Element]:
        """
        Parse from a local filesystem path.

        Args:
            file_path: Local file path (pdf/docx/pptx/xlsx/etc.)

        Returns:
            List of unstructured elements.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return partition(
            filename=str(path),
            strategy=self.options.strategy,
            infer_table_structure=self.options.infer_table_structure,
            languages=self.options.languages,
        )

    def parse_bytes(
        self,
        file_bytes: bytes,
        *,
        filetype: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> list[Element]:
        """
        Parse from in-memory bytes.

        For best auto-detection, pass `filename` (with extension) OR explicit `filetype`.
        `filetype` can be a MIME type like "application/pdf" or an extension like "pdf".

        Args:
            file_bytes: Raw file bytes.
            filetype: Optional. MIME type or extension hint.
            filename: Optional. Used by unstructured to infer file type.

        Returns:
            List of unstructured elements.
        """
        if not file_bytes:
            raise ValueError("Cannot parse empty file bytes")

        file_obj = BytesIO(file_bytes)
        file_obj.seek(0)

        # `partition` accepts `file=` with an optional `filetype=...`.
        # If filename is provided, we also pass it as metadata via `metadata_filename`.
        
        kwargs = {
            "file": file_obj,
            "metadata_filename": filename,
            "strategy": self.options.strategy,
            "infer_table_structure": self.options.infer_table_structure,
            "languages": self.options.languages,
        }
        
        # Only pass filetype if explicitly provided, to avoid conflicts with auto-detection
        if filetype:
            kwargs["filetype"] = filetype
            
        return partition(**kwargs)


