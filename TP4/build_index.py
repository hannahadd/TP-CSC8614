"""
build_index.py
Construit un index Chroma (persistant) à partir :
- d'emails .md dans data/emails/
- de PDF administratifs dans data/admin_pdfs/

Sortie :
- base Chroma dans chroma_db/
"""

import os
import glob
import shutil
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# Chemins relatifs au dossier TP4/ (car tu lances depuis TP4/)
DATA_DIR = "data"
EMAIL_DIR = os.path.join(DATA_DIR, "emails")
PDF_DIR = os.path.join(DATA_DIR, "admin_pdfs")

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "tp4_rag"

# Embeddings via Ollama
EMBEDDING_MODEL = "bge-m3"
PORT = "11435"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def load_emails(email_dir: str) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(glob.glob(os.path.join(email_dir, "*.md"))):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_type": "email",
                    "source": os.path.basename(path),
                    "path": path,
                },
            )
        )
    return docs


def load_pdfs(pdf_dir: str) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    docs: List[Document] = []
    for path in sorted(glob.glob(os.path.join(pdf_dir, "*.pdf"))):
        loader = PyPDFLoader(path)
        pages = loader.load()
        for p in pages:
            p.metadata["doc_type"] = "admin_pdf"
            p.metadata["source"] = os.path.basename(path)
            p.metadata["path"] = path
            docs.append(p)
    return docs


def main():
    if not os.path.isdir(EMAIL_DIR):
        raise FileNotFoundError(f"Dossier emails introuvable: {EMAIL_DIR}")
    if not os.path.isdir(PDF_DIR):
        raise FileNotFoundError(f"Dossier PDFs introuvable: {PDF_DIR}")

    email_docs = load_emails(EMAIL_DIR)
    pdf_docs = load_pdfs(PDF_DIR)
    docs = email_docs + pdf_docs

    print(f"[INFO] Emails chargés: {len(email_docs)}")
    print(f"[INFO] Pages PDF chargées: {len(pdf_docs)}")
    print(f"[INFO] Total documents bruts: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Total chunks: {len(chunks)}")

    if os.path.isdir(CHROMA_DIR):
        print(f"[WARN] {CHROMA_DIR} existe déjà. Suppression puis reconstruction.")
        shutil.rmtree(CHROMA_DIR)

    emb = OllamaEmbeddings(base_url=f"http://127.0.0.1:{PORT}", model=EMBEDDING_MODEL)

    _ = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print("[INFO] Index construit.")
    print(f"[DONE] Index persistant dans: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
