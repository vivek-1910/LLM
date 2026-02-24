#!/usr/bin/env python
"""
PDF Generator for LLM Assignment
Generates a comprehensive PDF with code and results
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Preformatted, Table, TableStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from datetime import datetime
import os
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.token import Token
import re

def create_title_page(story):
    """Create the title page"""
    styles = getSampleStyleSheet()
    
    # Add large top spacing for vertical centering
    story.append(Spacer(1, 2*inch))
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=36,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    story.append(Paragraph("LLM BANANA PROBLEM - 1", title_style))
    
    story.append(Spacer(1, 1.5*inch))
    
    # Student Info - Larger font
    normal_style = ParagraphStyle(
        'NormalCenter',
        parent=styles['Normal'],
        fontSize=16,
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    story.append(Paragraph("Name: <b>VIVEK GOWDA S</b>", normal_style))
    story.append(Paragraph("SRN: <b>PES1UG23AM355</b>", normal_style))
    story.append(Paragraph("Section: <b>F</b>", normal_style))


def add_section_header(story, title, styles):
    """Add a section header"""
    header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    story.append(Paragraph(title, header_style))
    story.append(Spacer(1, 0.15*inch))


def get_syntax_highlighted_code(code_content):
    """Get syntax highlighted code with color information"""
    lexer = PythonLexer()
    tokens = list(lexer.get_tokens(code_content))
    
    # Map token types to colors
    token_colors = {
        Token.Keyword: '#0000FF',           # Blue - keywords
        Token.Name.Builtin: '#0000FF',      # Blue - built-in functions
        Token.String: '#008000',             # Green - strings
        Token.Number: '#FF0000',             # Red - numbers
        Token.Comment: '#808080',            # Gray - comments
        Token.Operator: '#000000',           # Black - operators
        Token.Name: '#000000',               # Black - names
        Token.Name.Function: '#0000FF',      # Blue - function names
        Token.Literal.String: '#008000',     # Green - string literals
    }
    
    highlighted_lines = []
    current_line = []
    
    for token_type, value in tokens:
        if value == '\n':
            highlighted_lines.append(''.join(current_line))
            current_line = []
        else:
            # Find color for this token
            color = '#000000'  # Default to black
            for ttype, tcolor in token_colors.items():
                if token_type in ttype:
                    color = tcolor
                    break
            
            current_line.append(f'<font color="{color}">{value}</font>')
    
    if current_line:
        highlighted_lines.append(''.join(current_line))
    
    return '\n'.join(highlighted_lines)


def add_code_section(story, title, code_content, styles):
    """Add a code section with syntax highlighting"""
    styles = getSampleStyleSheet()
    
    # Section title
    subtitle_style = ParagraphStyle(
        'CodeTitle',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=8,
        spaceBefore=4,
        fontName='Helvetica-Bold'
    )
    story.append(Paragraph(title, subtitle_style))
    story.append(Spacer(1, 0.08*inch))
    
    # Code block style with syntax highlighting support
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontSize=7,
        fontName='Courier',
        leftIndent=12,
        rightIndent=12,
        textColor=colors.HexColor('#333333'),
        backColor=colors.HexColor('#f5f5f0'),
        borderColor=colors.HexColor('#d0d0d0'),
        borderWidth=1,
        borderPadding=8,
        spaceAfter=10,
        alignment=TA_CENTER
    )
    
    # Try to apply syntax highlighting
    try:
        highlighted = get_syntax_highlighted_code(code_content)
        lines = highlighted.split('\n')
    except:
        lines = code_content.split('\n')
    
    # Split into chunks
    chunk_size = 45
    for i in range(0, len(lines), chunk_size):
        chunk = '\n'.join(lines[i:i+chunk_size])
        try:
            story.append(Paragraph(chunk, code_style))
        except:
            # Fallback to Preformatted if Paragraph fails
            story.append(Preformatted(chunk, code_style))
        
        if i + chunk_size < len(lines):
            story.append(Spacer(1, 0.05*inch))


def add_results_section(story, title, image_path, image_width=6*inch):
    """Add a results section with image"""
    styles = getSampleStyleSheet()
    
    # Section title
    subtitle_style = ParagraphStyle(
        'ResultTitle',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=8,
        spaceBefore=4,
        fontName='Helvetica-Bold'
    )
    story.append(Paragraph(title, subtitle_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Add image if it exists
    if os.path.exists(image_path):
        try:
            img = Image(image_path, width=image_width, height=3.5*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
        except Exception as e:
            print(f"Warning: Could not add image {image_path}: {e}")
    else:
        print(f"Warning: Image not found at {image_path}")
    
    story.append(Spacer(1, 0.1*inch))


def main():
    """Main function to generate PDF"""
    
    # Create PDF with SRN in filename
    pdf_filename = "/Volumes/T7Shield/LLM/PES1UG23AM355_LLM_BANANA_PROBLEM_ASSIGNMENT.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=0.4*inch,
        leftMargin=0.4*inch,
        topMargin=0.4*inch,
        bottomMargin=0.4*inch
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Title Page
    create_title_page(story)
    story.append(PageBreak())
    
    # Read code files
    print("Reading code files...")
    with open('/Volumes/T7Shield/LLM/assignment/ANN/Binary/binary_classification.py', 'r') as f:
        binary_code = f.read()
    
    with open('/Volumes/T7Shield/LLM/assignment/ANN/Multiclass/multiclass_classification.py', 'r') as f:
        multiclass_code = f.read()
    
    with open('/Volumes/T7Shield/LLM/assignment/LR/logistic_regression.py', 'r') as f:
        lr_code = f.read()
    
    # Define image paths
    binary_image = '/Volumes/T7Shield/LLM/assignment/ANN/Binary/results.png'
    multiclass_image = '/Volumes/T7Shield/LLM/assignment/ANN/Multiclass/results.png'
    lr_image = '/Volumes/T7Shield/LLM/assignment/LR/results.png'
    
    # Section 1: Binary Classification
    add_section_header(story, "1. BINARY CLASSIFICATION", styles)
    add_code_section(story, "Heart Disease Prediction Code", binary_code, styles)
    story.append(PageBreak())
    
    add_results_section(story, "Results & Visualizations", binary_image)
    story.append(PageBreak())
    
    # Section 2: Multiclass Classification
    add_section_header(story, "2. MULTI-CLASS CLASSIFICATION", styles)
    add_code_section(story, "Iris Flower Classification Code", multiclass_code, styles)
    story.append(PageBreak())
    
    add_results_section(story, "Results & Visualizations", multiclass_image)
    story.append(PageBreak())
    
    # Section 3: Logistic Regression
    add_section_header(story, "3. LOGISTIC REGRESSION", styles)
    add_code_section(story, "Binary Classification Code", lr_code, styles)
    story.append(PageBreak())
    
    add_results_section(story, "Results & Visualizations", lr_image)
    
    # Build PDF
    print("Building PDF...")
    doc.build(story)
    print(f"✅ PDF generated successfully: {pdf_filename}")
    print(f"📄 File size: {os.path.getsize(pdf_filename) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
