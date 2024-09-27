from PyPDF2 import PdfReader
from llama_index.core import Document

class PDFLoader:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths

    def load_pdfs(self):
        documents = []
        for path in self.pdf_paths:
            text = self.extract_text_from_pdf(path)
            documents.append(Document(text=text))
        return documents

    def extract_text_from_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text