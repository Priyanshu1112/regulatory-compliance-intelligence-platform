from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    """
    Handles chunking of document texts into smaller, configurable blocks.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_pages(self, pages: list[dict]) -> list[dict]:
        """
        Splits pages into smaller text chunks while preserving and extending metadata.

        Args:
            pages: A list of page dicts from DocumentLoader.

        Returns:
            A list of chunk dicts:
            {
                "text": str,
                "metadata": {
                    "filename": str,
                    "page": int,
                    "upload_date": str,
                    "version": str,
                    "file_size": int,
                    "chunk_index": int
                }
            }
        """
        chunks = []
        chunk_index = 0
        for page in pages:
            text = page["text"]
            metadata = page["metadata"]
            if not text.strip():
                continue

            page_chunks = self.splitter.split_text(text)
            for page_chunk in page_chunks:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index
                
                chunks.append({
                    "text": page_chunk,
                    "metadata": chunk_metadata
                })
                chunk_index += 1

        return chunks
