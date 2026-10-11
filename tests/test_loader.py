import pytest
from app.rag.loader import load_pdf_documents


def test_load_pdf_documents_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_pdf_documents("/tmp/does_not_exist_12345.pdf")
