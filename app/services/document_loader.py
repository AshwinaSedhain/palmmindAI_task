import fitz  # PyMuPDF
from fastapi import UploadFile


ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def validate_file_extension(filename: str):
    ext = filename[filename.rfind("."):].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type. Only PDF and TXT allowed.")
    return ext


async def extract_text(file: UploadFile):
    filename = file.filename or ""
    ext = validate_file_extension(filename)
    
    content = await file.read()
    if not content:
        raise ValueError("File is empty")
    
    if ext == ".txt":
        return extract_txt(content)
    else:
        return extract_pdf(content, filename)


def extract_txt(data: bytes):
    try:
        text = data.decode("utf-8")
    except:
        text = data.decode("latin-1")
    
    text = text.strip()
    if not text:
        raise ValueError("No text content found")
    return text


def extract_pdf(data: bytes, filename: str):
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}")
    
    pages = []
    for i in range(len(doc)):
        page_text = doc.load_page(i).get_text()
        if page_text.strip():
            pages.append(page_text.strip())
    
    doc.close()
    
    if not pages:
        raise ValueError("No text found in PDF")
    
    return "\n\n".join(pages)
