import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# French model
EMBEDDING_MODEL_TYPE = "CamembertEmbeddings"
EMBEDDING_MODEL_NAME = "dangvantuan/sentence-camembert-large"

# HuggingFaceEmbeddings
#EMBEDDING_MODEL_NAME = "dangvantuan/sentence-camembert-base"
#EMBEDDING_MODEL_TYPE = "HuggingFaceEmbeddings"
#EMBEDDING_MODEL_NAME = "inokufu/flaubert-base-uncased-xnli-sts" # aucun résultat
#EMBEDDING_MODEL_NAME = "Sahajtomar/french_semantic" # très faible
#EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"  # encore assez faible
#EMBEDDING_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"
#EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-multilingual"
#EMBEDDING_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
#EMBEDDING_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
#EMBEDDING_MODEL_NAME = "dangvantuan/sentence-camembert-large"  # there is a loading issue with that model
# Smaller funtionnal HuggingFaceEmbeddings model
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# HuggingFaceInstructEmbeddings  (Really good for english)
# EMBEDDING_MODEL_TYPE = "HuggingFaceInstructEmbeddings"
# EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"



# Select the Model ID and model_basename for the LLM
# load the LLM for generating Natural Language responses

# MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
# MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"

# for HF models
# MODEL_ID = "TheBloke/vicuna-7B-1.1-HF"
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/guanaco-7B-HF"
# MODEL_ID = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
# alongside will 100% create OOM on 24GB cards.
# llm = load_model(device_type, model_id=model_id)

# for GPTQ (quantized) models
# MODEL_ID = "TheBloke/Nous-Hermes-13B-GPTQ"
# MODEL_BASENAME = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
# MODEL_ID = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" # Requires
# ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
# MODEL_ID = "TheBloke/wizardLM-7B-GPTQ"
# MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
# MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"

# for GGML (quantized cpu+gpu+mps) models - check if they support llama.cpp
# MODEL_ID = "TheBloke/wizard-vicuna-13B-GGML"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
MODEL_ID = "TheBloke/orca_mini_3B-GGML"
MODEL_BASENAME = "orca-mini-3b.ggmlv3.q4_0.bin"


