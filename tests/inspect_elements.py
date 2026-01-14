"""
Inspect and export unstructured elements to JSON.

Usage:
    python inspect_elements.py /path/to/document.pdf
    python inspect_elements.py /path/to/document.pdf --output custom_output.json

This script:
1. Parses the document using doc_parser
2. Extracts ALL attributes from each element
3. Writes complete element data to JSON file (filename_parsed_elements.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from unstructured.staging.base import elements_to_dicts

from doc_parser import UnstructuredDocParser, ParseOptions


def extract_all_element_data(element: Any) -> dict[str, Any]:
    """
    Extract ALL attributes from an unstructured Element.
    
    Uses multiple methods to ensure we capture everything:
    - elements_to_dicts (unstructured's official serialization)
    - Direct attribute access
    - __dict__ inspection
    """
    # Method 1: Use unstructured's official serialization (most reliable)
    try:
        element_dict = elements_to_dicts([element])[0]
    except Exception as e:
        element_dict = {"_serialization_error": str(e)}
    
    # Method 2: Also capture any additional attributes via __dict__
    try:
        element_vars = vars(element)
        # Merge any attributes not in the dict
        for key, value in element_vars.items():
            if key not in element_dict:
                element_dict[f"_attr_{key}"] = str(value) if not isinstance(value, (dict, list)) else value
    except Exception:
        pass
    
    # Method 3: Direct attribute access for common ones
    common_attrs = ["text", "type", "category", "metadata"]
    for attr in common_attrs:
        if hasattr(element, attr):
            attr_value = getattr(element, attr)
            # Only add if not already captured
            if attr not in element_dict:
                element_dict[attr] = attr_value
    
    return element_dict


def inspect_document(file_path: str, output_path: str | None = None) -> None:
    """
    Parse document and export all element data to JSON.
    
    Args:
        file_path: Path to document file
        output_path: Optional custom output path. Defaults to {filename}_parsed_elements.json
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"📄 Parsing document: {path.name}")
    print(f"   Path: {path}")
    print(f"   Size: {path.stat().st_size / 1024:.2f} KB")
    
    # Parse document
    try:
        parser = UnstructuredDocParser(
            options=ParseOptions(
                strategy="hi_res",  # Best quality parsing
                infer_table_structure=True,
            )
        )
        elements = parser.parse_path(path)
        print(f"✅ Parsed successfully: {len(elements)} elements extracted")
    except Exception as e:
        print(f"❌ Error parsing document: {e}")
        sys.exit(1)
    
    # Extract all data from each element
    print("🔍 Extracting element data...")
    elements_data = []
    
    for idx, element in enumerate(elements):
        try:
            element_dict = extract_all_element_data(element)
            # Add index for reference
            element_dict["_element_index"] = idx
            elements_data.append(element_dict)
        except Exception as e:
            print(f"⚠️  Warning: Failed to extract element {idx}: {e}")
            elements_data.append({
                "_element_index": idx,
                "_error": str(e),
                "_element_type": str(type(element).__name__)
            })
    
    # Determine output path
    if output_path is None:
        output_path = path.stem + "_parsed_elements.json"
    output_path_obj = Path(output_path)
    
    # Create summary stats
    element_types = {}
    for elem in elements_data:
        elem_type = elem.get("type") or elem.get("category") or elem.get("_element_type", "Unknown")
        element_types[elem_type] = element_types.get(elem_type, 0) + 1
    
    # Prepare final output
    output_data = {
        "source_file": str(path),
        "source_filename": path.name,
        "total_elements": len(elements),
        "element_type_counts": element_types,
        "elements": elements_data
    }
    
    # Write to JSON
    try:
        with open(output_path_obj, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Exported to: {output_path_obj}")
        print(f"   Total elements: {len(elements)}")
        print(f"   Element types found: {len(element_types)}")
        print("\n📊 Element type breakdown:")
        for elem_type, count in sorted(element_types.items(), key=lambda x: -x[1]):
            print(f"   - {elem_type}: {count}")
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)
    
    print(f"\n🎉 Inspection complete! Open {output_path_obj} to see all element data.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect unstructured document elements and export to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_elements.py document.pdf
  python inspect_elements.py document.pdf --output custom_name.json
  python inspect_elements.py /path/to/presentation.pptx
        """
    )
    
    parser.add_argument(
        "file_path",
        help="Path to document file (PDF, DOCX, PPTX, XLSX, etc.)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: {filename}_parsed_elements.json)"
    )
    
    args = parser.parse_args()
    inspect_document(args.file_path, args.output)


if __name__ == "__main__":
    main()

