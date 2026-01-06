from src.Agent.functions import document_ocr

if __name__ == "__main__":
    test_document_path = "data/test/test_pdf.pdf"
    ocr_result = document_ocr(test_document_path)
    print(ocr_result)