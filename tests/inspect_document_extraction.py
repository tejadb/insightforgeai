"""
Document Extraction Inspection Script

Processes a document and saves:
1. All elements as complete JSON (nothing skipped)
2. The doc_context string (what we give to LLM)
3. All chunks as JSON (only content/text)

Usage (from project root):
    python tests/inspect_document_extraction.py "tests/Assignments/Corporate Finance  Assignment ss.pdf"
    
Or from tests folder:
    python inspect_document_extraction.py "Assignments/Corporate Finance  Assignment ss.pdf"
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unstructured.staging.base import elements_to_dicts
from doc_parser import UnstructuredDocParser, ParseOptions
from elements_processor import doc_context
from chunking import chunk_document_elements, chunk_to_text, get_chunking_config


def save_elements_json(elements, output_path: Path):
    """Save all elements as complete JSON (nothing skipped)."""
    # Convert elements to dicts (includes all attributes)
    elements_dicts = elements_to_dicts(elements)
    
    # Save with pretty formatting
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(elements_dicts, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(elements_dicts)} elements to {output_path}")
    return elements_dicts


def save_doc_context(context_text: str, output_path: Path):
    """Save the doc_context string."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(context_text)
    
    print(f"✅ Saved doc_context ({len(context_text)} chars) to {output_path}")


def save_chunks_json(chunked_elements, output_path: Path):
    """Save all chunks as JSON (only content/text)."""
    chunks_data = []
    
    for idx, chunk_elem in enumerate(chunked_elements):
        chunk_text = chunk_to_text(chunk_elem)
        if not chunk_text.strip():
            continue
        
        chunks_data.append({
            "chunk_index": idx,
            "content": chunk_text,
            "content_length": len(chunk_text),
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(chunks_data)} chunks to {output_path}")
    return chunks_data


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_document_extraction.py <file_path>")
        print("  File path can be relative to tests/ folder or absolute")
        sys.exit(1)
    
    # Get script directory (tests folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Handle file path (can be relative to tests/ or absolute)
    file_path_arg = sys.argv[1]
    file_path = Path(file_path_arg)
    
    # If not absolute, try relative to tests folder first, then project root
    if not file_path.is_absolute():
        test_relative = script_dir / file_path_arg
        if test_relative.exists():
            file_path = test_relative
        else:
            file_path = project_root / file_path_arg
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    # Create output directory in tests folder
    output_dir = script_dir / "extraction_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base filename for outputs (with timestamp to preserve old files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = file_path.stem
    base_name_with_timestamp = f"{base_name}_{timestamp}"
    
    print(f"📄 Processing: {file_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📅 Timestamp: {timestamp}\n")
    
    # 1. Parse document
    print("Step 1: Parsing document...")
    parser = UnstructuredDocParser(options=ParseOptions(strategy="hi_res"))
    elements = parser.parse_path(file_path)
    print(f"✅ Parsed {len(elements)} elements\n")
    
    # 2. Save all elements as JSON
    print("Step 2: Saving all elements...")
    elements_json_path = output_dir / f"{base_name_with_timestamp}_elements.json"
    elements_dicts = save_elements_json(elements, elements_json_path)
    
    # Print element type summary
    element_types = {}
    for elem in elements:
        elem_type = getattr(elem, "type", None) or getattr(elem, "category", "Unknown")
        element_types[elem_type] = element_types.get(elem_type, 0) + 1
    print(f"   Element types: {dict(sorted(element_types.items()))}\n")
    
    # 3. Generate and save doc_context
    print("Step 3: Generating doc_context...")
    context_text = doc_context(elements, filename=file_path.name)
    context_path = output_dir / f"{base_name_with_timestamp}_doc_context.txt"
    save_doc_context(context_text, context_path)
    print()
    
    # 4. Chunk document
    print("Step 4: Chunking document...")
    chunking_config = get_chunking_config()
    print(f"   Config: {chunking_config}")
    chunked_elements = chunk_document_elements(elements, **chunking_config)
    print(f"✅ Created {len(chunked_elements)} chunks\n")
    
    # 5. Save chunks as JSON
    print("Step 5: Saving chunks...")
    chunks_json_path = output_dir / f"{base_name_with_timestamp}_chunks.json"
    chunks_data = save_chunks_json(chunked_elements, chunks_json_path)
    
    # Print chunk summary
    print(f"\n📊 Summary:")
    print(f"   Total elements: {len(elements)}")
    print(f"   Total chunks: {len(chunks_data)}")
    print(f"   Average chunk size: {sum(c['content_length'] for c in chunks_data) / len(chunks_data) if chunks_data else 0:.0f} chars")
    print(f"   Min chunk size: {min(c['content_length'] for c in chunks_data) if chunks_data else 0} chars")
    print(f"   Max chunk size: {max(c['content_length'] for c in chunks_data) if chunks_data else 0} chars")
    
    print(f"\n✅ All outputs saved to: {output_dir}/")
    print(f"   - {base_name_with_timestamp}_elements.json (all elements)")
    print(f"   - {base_name_with_timestamp}_doc_context.txt (LLM context)")
    print(f"   - {base_name_with_timestamp}_chunks.json (chunks content)")
    print(f"\n💡 Note: Files are timestamped to preserve old outputs for comparison.")


if __name__ == "__main__":
    main()

