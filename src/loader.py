import os
from datetime import datetime
import pypdf

class DocumentLoader:
    """
    Handles PDF document loading and text extraction.
    """
    def __init__(self):
        pass

    def load_pdf(self, file_path: str) -> list[dict]:
        """
        Loads a PDF document page by page.
        
        Args:
            file_path: The absolute or relative path to the PDF file.
            
        Returns:
            A list of dicts, each representing a page:
            {
                "text": str,
                "metadata": {
                    "filename": str,
                    "page": int (1-indexed),
                    "upload_date": str (ISO 8601),
                    "version": str,
                    "file_size": int
                }
            }
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        filename = os.path.basename(file_path)
        upload_date = datetime.now().isoformat()
        file_size = os.path.getsize(file_path)

        pages = []
        try:
            reader = pypdf.PdfReader(file_path)
            for index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append({
                    "text": text,
                    "metadata": {
                        "filename": filename,
                        "page": index + 1,
                        "upload_date": upload_date,
                        "version": "1.0",
                        "file_size": file_size
                    }
                })
        except Exception as e:
            raise ValueError(f"Failed to parse PDF file {filename}: {str(e)}")

        return pages
