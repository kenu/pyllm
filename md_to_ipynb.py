#!/usr/bin/env python3
"""
Convert Markdown (.md) files to Jupyter Notebook (.ipynb) format.
Compatible with Google Colab.
"""

import json
import re
import sys
from pathlib import Path


def parse_markdown_to_cells(md_content):
    """
    Parse markdown content into notebook cells.
    Code blocks become code cells, everything else becomes markdown cells.
    """
    cells = []
    lines = md_content.split('\n')
    
    current_cell = []
    current_type = 'markdown'
    in_code_block = False
    code_language = 'python'
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for code block start
        if line.startswith('```'):
            # Save current cell if it has content
            if current_cell and current_type == 'markdown':
                cells.append({
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': current_cell
                })
                current_cell = []
            
            # Extract language if specified
            lang_match = re.match(r'```(\w+)?', line)
            if lang_match and lang_match.group(1):
                code_language = lang_match.group(1)
            
            in_code_block = True
            current_type = 'code'
            i += 1
            
            # Collect code block content
            code_content = []
            while i < len(lines) and not lines[i].startswith('```'):
                code_content.append(lines[i] + '\n')
                i += 1
            
            # Create code cell (only if language is python or similar)
            if code_language.lower() in ['python', 'py', 'python3']:
                cells.append({
                    'cell_type': 'code',
                    'metadata': {},
                    'source': code_content,
                    'outputs': [],
                    'execution_count': None
                })
            else:
                # For non-Python code, keep as markdown
                markdown_content = [f'```{code_language}\n'] + code_content + ['```\n']
                cells.append({
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': markdown_content
                })
            
            current_cell = []
            current_type = 'markdown'
            in_code_block = False
            
        else:
            # Regular markdown content
            if current_type == 'markdown':
                current_cell.append(line + '\n')
        
        i += 1
    
    # Add remaining content as markdown cell
    if current_cell:
        cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': current_cell
        })
    
    return cells


def create_notebook(cells):
    """Create a Jupyter notebook structure."""
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.10.12'
            },
            'colab': {
                'provenance': []
            }
        },
        'nbformat': 4,
        'nbformat_minor': 0
    }
    return notebook


def convert_md_to_ipynb(md_file, output_file=None):
    """Convert a markdown file to Jupyter notebook format."""
    md_path = Path(md_file)
    
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_file}")
    
    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Parse markdown into cells
    cells = parse_markdown_to_cells(md_content)
    
    # Create notebook structure
    notebook = create_notebook(cells)
    
    # Determine output file
    if output_file is None:
        output_file = md_path.with_suffix('.ipynb')
    
    # Write notebook
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✓ Converted: {md_file} → {output_file}")
    return output_file


def main():
    """Main function to handle command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python md_to_ipynb.py <markdown_file.md> [output_file.ipynb]")
        print("\nExample:")
        print("  python md_to_ipynb.py example.md")
        print("  python md_to_ipynb.py example.md output.ipynb")
        sys.exit(1)
    
    md_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        convert_md_to_ipynb(md_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
