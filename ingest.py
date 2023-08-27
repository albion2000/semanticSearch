import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from sentence_transformers import SentenceTransformer

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_TYPE,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)



def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory
    all_files = os.listdir(source_dir)
    paths = []
    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1]
        source_file_path = os.path.join(source_dir, file_path)
        if file_extension in DOCUMENT_MAP.keys():
            paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs

class ST_with_ed(SentenceTransformer):
    def __init__(self, model_name):
        super().__init__(model_name)
    def embed_documents(settings,docs):
        out = []
        for setting in settings:
            print(setting)
        for doc in docs:
            print(doc)
            print("*****")
#        for doc in docs:
#            out.append(super().encode(doc))
        return super().encode(docs)
        

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)



def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")


    # Create embeddings
    if (EMBEDDING_MODEL_TYPE == "HuggingFaceInstructEmbeddings"):
        embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME,model_kwargs={"device": device_type})
	    # change the embedding type here if you are running into issues.
	    # These are much smaller embeddings and will work for most appications
	    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
	    # run_localGPT.py file.
    elif (EMBEDDING_MODEL_TYPE == "HuggingFaceEmbeddings"):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    elif (EMBEDDING_MODEL_TYPE == "CamembertEmbeddings"):
#        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME) 
        embeddings = ST_with_ed(model_name=EMBEDDING_MODEL_NAME)
        embeddings.max_seq_length=512
    else:
        print("undefined or unexpected EMBEDDING_MODEL_TYPE")
        return

# https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.chroma.Chroma.html    fucking hard to find

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None
    # BELL  
    print('\a')


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
