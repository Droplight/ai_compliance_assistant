from pypdf import PdfReader
import os

def load_pdfs(folder_path="data"):
    all_text = ""

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            reader = PdfReader(pdf_path)

            for page in reader.pages:
                text = page.extract_text() or ""
                all_text += text + "\n"

    return all_text

if __name__ == "__main__":
    text = load_pdfs()
    print("Loaded PDF characters:", len(text))
