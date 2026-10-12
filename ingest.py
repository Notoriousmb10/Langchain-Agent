from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import hashlib
from langchain_community.document_loaders import TextLoader
from common import data_dir, manifest_path, vector_store


def ingest():

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        add_start_index=True,
    )

    files = sorted(data_dir.glob("*.txt"))

    for file_path in files:
        source = file_path.as_posix()

        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

        if manifest.get(source) == file_hash:
            print(f"Skipping {source}, already ingested.")
            continue

        print(f"Ingesting {source}...")

        vector_store.delete(where={"source": source})

        loader = TextLoader(source, encoding="utf-8")

        documents = loader.load()

        chunks = text_splitter.split_documents(documents)

        chunk_ids = [f"{source}::chunk::{i}" for i in range(len(chunks))]

        if chunks:
            vector_store.add_documents(
                documents=chunks,
                ids=chunk_ids,
            )

        manifest[source] = file_hash

        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


print("Indexing complete.")
