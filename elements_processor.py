"""
Elements processor - converts unstructured elements to simple text format.

Simple post-processing: takes elements and creates a single combined text string.
"""

from __future__ import annotations

import html
from typing import Iterable

from unstructured.documents.elements import Element, Table


def doc_context(elements: Iterable[Element], filename: str = "") -> str:
    """
    Combine all elements into a single text string with XML-style page markers.
    
    Format:
    <doc_name>
    <page 1 start>
    element 1 text
    element 2 text
    <page 1 end>
    <page 2 start>
    ...
    <page 2 end>
    </doc_name>
    
    Args:
        elements: List of unstructured Element objects
        filename: Document filename to use in doc_name tag
        
    Returns:
        Single combined text string with page markers wrapped in doc_name tags
    """
    # Convert to list if needed and check if empty
    elements_list = list(elements)
    if not elements_list:
        doc_name = filename.strip() if filename else "document"
        return f"<{doc_name}>\n</{doc_name}>"
    
    parts: list[str] = []
    
    # Get document name (use filename or default) and escape XML special characters
    doc_name_raw = filename.strip() if filename else "document"
    doc_name = html.escape(doc_name_raw)  # Escape <, >, & to prevent XML injection
    
    # Start document wrapper
    parts.append(f"<{doc_name}>")
    parts.append("")
    
    # Group elements by page number
    current_page = None
    
    for element in elements_list:
        # Get page number from metadata
        page_num = None
        if hasattr(element, "metadata") and element.metadata:
            page_num = getattr(element.metadata, "page_number", None)
        
        # Get element text
        elem_text = getattr(element, "text", "") or ""
        
        # Special handling for tables: prefer HTML structure for better retention
        if isinstance(element, Table) or getattr(element, "type", None) == "Table":
            table_html = None
            if hasattr(element, "metadata") and element.metadata:
                table_html = getattr(element.metadata, "text_as_html", None)
            elem_text_to_use = table_html or elem_text
        else:
            elem_text_to_use = elem_text
        
        # Skip if no text/HTML (like PageBreak elements)
        if not elem_text_to_use.strip():
            continue
        
        # If page number changed, close previous page and start new one
        if page_num is not None and page_num != current_page:
            # Close previous page if exists
            if current_page is not None:
                parts.append(f"<page {current_page} end>")
                parts.append("")
            
            # Start new page
            current_page = page_num
            parts.append(f"<page {current_page} start>")
        
        # Add element text to current page
        if current_page is not None:
            parts.append(elem_text_to_use)
        else:
            # If no page number, just add text (shouldn't happen normally)
            parts.append(elem_text_to_use)
    
    # Close last page if exists
    if current_page is not None:
        parts.append(f"<page {current_page} end>")
    
    # Close document wrapper
    parts.append("")
    parts.append(f"</{doc_name}>")
    
    return "\n".join(parts)

