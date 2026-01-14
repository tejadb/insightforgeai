"""
Chunking utilities for RAG.

Uses unstructured.io's chunking functions to split documents into semantic chunks
while preserving structure (tables, sections, etc.).
"""

import os
import re
from typing import List, Dict, Any, Optional
from unstructured.documents.elements import Element, Table, Title
from unstructured.chunking.title import chunk_by_title


def chunk_document_elements(
    elements: List[Element],
    strategy: str = "by_title",
    max_characters: int = 1000,
    new_after_n_chars: int = 500,
    overlap: int = 50,
    overlap_all: bool = False,
    multipage_sections: bool = True,
    combine_text_under_n_chars: int = 500,
) -> List[Element]:
    """
    Chunk document elements using unstructured.io's chunking strategies.
    
    Simple approach: Separate tables from other elements, chunk others normally,
    then append tables at the end. This preserves table integrity while keeping
    the code simple and maintainable.
    
    Args:
        elements: List of parsed Element objects from unstructured.io
        strategy: Chunking strategy ("by_title", "basic")
        max_characters: Hard limit on chunk size (1000)
        new_after_n_chars: Soft limit - start new chunk after this (500)
        overlap: Character overlap between chunks (50)
        overlap_all: Apply overlap to all chunks (False - only split elements)
        multipage_sections: Allow sections to span pages (for by_title strategy)
        combine_text_under_n_chars: Combine small sections (for by_title strategy) (500)
    
    Returns:
        List of chunked Element objects (CompositeElement, Table, etc.)
        Tables are always kept as single atomic chunks, never split.
        Tables are appended after text chunks.
    """
    if not elements:
        return []
    
    # STEP 1: Separate tables from other elements
    # Tables must NEVER be split, so we handle them separately
    tables = []
    other_elements = []
    
    for elem in elements:
        if isinstance(elem, Table) or getattr(elem, "type", None) == "Table":
            tables.append(elem)  # Keep tables separate - they're atomic
        else:
            other_elements.append(elem)
    
    # STEP 2: Chunk non-table elements normally using unstructured's standard functions
    chunked_others = []
    
    if strategy == "by_title":
        chunked_others = chunk_by_title(
            other_elements,
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars,
            overlap=overlap,
            overlap_all=overlap_all,
            multipage_sections=multipage_sections,
            combine_text_under_n_chars=combine_text_under_n_chars,
        )
    elif strategy == "basic":
        from unstructured.chunking.basic import chunk_elements
        chunked_others = chunk_elements(
            other_elements,
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars,
            overlap=overlap,
            overlap_all=overlap_all,
        )
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")
    
    # STEP 3: Combine chunked elements with tables
    # Simple approach: text chunks first, then tables
    # chunk_index will reflect this order (0, 1, 2, ... for text chunks, then tables)
    all_chunks = list(chunked_others)
    all_chunks.extend(tables)  # Append tables at the end
    
    return all_chunks


def extract_chunk_metadata(
    chunk_element: Element,
    original_elements: List[Element],
) -> Dict[str, Any]:
    """
    Extract metadata from a chunk element for storage.
    
    Args:
        chunk_element: The chunked Element (CompositeElement, Table, etc.)
        original_elements: Original elements list (for finding element IDs)
    
    Returns:
        Dictionary with metadata: page_start, page_end, section_heading, element_ids, element_types
    """
    metadata: Dict[str, Any] = {
        "element_types": [],
        "element_ids": [],
    }
    
    # Get page numbers from chunk element's metadata
    page_numbers = []
    
    if hasattr(chunk_element, "metadata") and chunk_element.metadata:
        page_num = getattr(chunk_element.metadata, "page_number", None)
        if page_num:
            page_numbers.append(page_num)
    
    # For CompositeElement, check constituent elements
    if hasattr(chunk_element, "elements"):
        for elem in chunk_element.elements:
            if hasattr(elem, "metadata") and elem.metadata:
                page_num = getattr(elem.metadata, "page_number", None)
                if page_num:
                    page_numbers.append(page_num)
            
            # Collect element types
            elem_type = getattr(elem, "type", None) or getattr(elem, "category", "Unknown")
            if elem_type not in metadata["element_types"]:
                metadata["element_types"].append(elem_type)
            
            # Collect element IDs
            elem_id = getattr(elem, "id", None) or getattr(elem, "element_id", None)
            if elem_id:
                metadata["element_ids"].append(str(elem_id))
    
    # Set page range
    if page_numbers:
        metadata["page_start"] = min(page_numbers)
        metadata["page_end"] = max(page_numbers)
    else:
        metadata["page_start"] = None
        metadata["page_end"] = None
    
    # Extract section heading (look for Title elements before this chunk)
    section_heading = None
    chunk_text = getattr(chunk_element, "text", "") or ""
    
    # Try to find a Title element in the chunk or preceding elements
    if hasattr(chunk_element, "elements"):
        for elem in chunk_element.elements:
            if isinstance(elem, Title) or getattr(elem, "type", None) == "Title":
                section_heading = getattr(elem, "text", "").strip()
                break
    
    # If no title in chunk, look for "CHAPTER-X" pattern in text
    if not section_heading:
        chapter_match = re.search(r"CHAPTER[-\s]*\d+", chunk_text, re.IGNORECASE)
        if chapter_match:
            section_heading = chapter_match.group(0)
    
    metadata["section_heading"] = section_heading
    
    # Special handling for Table elements
    if isinstance(chunk_element, Table) or getattr(chunk_element, "type", None) == "Table":
        metadata["element_types"] = ["Table"]
        
        # Store table HTML if available
        if hasattr(chunk_element, "metadata") and chunk_element.metadata:
            text_as_html = getattr(chunk_element.metadata, "text_as_html", None)
            if text_as_html:
                metadata["table_html"] = text_as_html
                # Count rows from HTML
                row_count = text_as_html.count("<tr>")
                metadata["row_count"] = row_count
    
    return metadata


def chunk_to_text(chunk_element: Element) -> str:
    """
    Convert a chunk Element to text string for embedding.
    
    For tables, uses HTML structure (text_as_html) to preserve table structure.
    For other elements, uses plain text.
    
    Args:
        chunk_element: Chunked Element (CompositeElement, Table, etc.)
    
    Returns:
        Text representation of the chunk (HTML for tables, plain text for others)
    """
    # For Table elements, use HTML structure if available (preserves table structure)
    if isinstance(chunk_element, Table) or getattr(chunk_element, "type", None) == "Table":
        # Try to get HTML structure first (better for embeddings and LLM understanding)
        if hasattr(chunk_element, "metadata") and chunk_element.metadata:
            html = getattr(chunk_element.metadata, "text_as_html", None)
            if html:
                return html  # Use HTML structure - preserves table layout
        
        # Fallback to plain text if HTML not available
        text = getattr(chunk_element, "text", "") or ""
        return text.strip()
    
    # For CompositeElement, combine all text
    if hasattr(chunk_element, "elements"):
        parts = []
        for elem in chunk_element.elements:
            elem_text = getattr(elem, "text", "") or ""
            if elem_text.strip():
                parts.append(elem_text.strip())
        if parts:
            return "\n\n".join(parts)
    
    # For other elements, use plain text
    text = getattr(chunk_element, "text", "") or ""
    return text.strip()


def get_chunking_config() -> Dict[str, Any]:
    """
    Load chunking configuration from environment variables.
    
    Default values:
    - max_characters: 1000 (hard limit)
    - new_after_n_chars: 500 (soft limit - chunks target ~500 chars)
    - overlap: 50 (moderate overlap for context)
    - overlap_all: False (only overlap split elements, not all chunks)
    - combine_text_under_n_chars: 500 (combine small sections)
    
    Returns:
        Dictionary with chunking parameters
    """
    return {
        "strategy": os.getenv("CHUNK_STRATEGY", "by_title"),
        "max_characters": int(os.getenv("CHUNK_MAX_CHARACTERS", "1000")),
        "new_after_n_chars": int(os.getenv("CHUNK_NEW_AFTER_N_CHARS", "500")),
        "overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
        "overlap_all": os.getenv("CHUNK_OVERLAP_ALL", "false").lower() == "true",
        "multipage_sections": os.getenv("CHUNK_MULTIPAGE_SECTIONS", "true").lower() == "true",
        "combine_text_under_n_chars": int(os.getenv("CHUNK_COMBINE_UNDER_N_CHARS", "500")),
    }

