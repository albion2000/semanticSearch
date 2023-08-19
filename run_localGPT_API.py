import logging
import os
import shutil
import subprocess
import torch

#from auto_gptq import AutoGPTQForCausalLM
from flask import Flask, jsonify, request
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings

# from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.llms import HuggingFacePipeline
#from run_localGPT import load_model

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
#from transformers import (
#    AutoModelForCausalLM,
#    AutoTokenizer,
#    GenerationConfig,
#    LlamaForCausalLM,
#    LlamaTokenizer,
#    pipeline,
#)

from werkzeug.utils import secure_filename

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY
#, MODEL_ID, MODEL_BASENAME

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
SHOW_SOURCES = True
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
#if os.path.exists(PERSIST_DIRECTORY):
#    try:
#        shutil.rmtree(PERSIST_DIRECTORY)
#    except OSError as e:
#        print(f"Error: {e.filename} - {e.strerror}.")
#else:
#    print("The directory does not exist")

#run_langest_commands = ["python", "ingest.py"]
#if DEVICE_TYPE == "cpu":
#    run_langest_commands.append("--device_type")
#    run_langest_commands.append(DEVICE_TYPE)

#result = subprocess.run(run_langest_commands, capture_output=True)
#if result.returncode != 0:
#    raise FileNotFoundError(
#        "No files were found inside SOURCE_DOCUMENTS, please put a starter file inside before starting the API!"
#    )

# load the vectorstore
DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)

# similarity search is the default, k=4 results is the default
#RETRIEVER = DB.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 10, 'score_threshold': 0.3})
RETRIEVER = DB.as_retriever(search_type="mmr", search_kwargs={'k': 20})

#LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)

#QA = RetrievalQA.from_chain_type(    llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=SHOW_SOURCES )

app = Flask(__name__)

@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
#    global QA
    global RETRIEVER
    user_prompt = request.form.get("user_prompt")
    if user_prompt:
        # print(f'User Prompt: {user_prompt}')
        # Get the answer from the chain
	# print(RETRIEVER.get_relevant_documents(user_prompt))
        
        docs = RETRIEVER.get_relevant_documents(user_prompt)
#        res = QA(user_prompt)
#        answer, docs = res["result"], res["source_documents"]

        prompt_response_dict = {
            "Prompt": user_prompt,
#            "Answer": answer,
            "Answer": "What did you expect ?",
        }

        prompt_response_dict["Sources"] = []
        for document in docs:
            prompt_response_dict["Sources"].append(
                (os.path.basename(str(document.metadata["source"])), str(document.page_content))
            )

        return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 400


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    app.run(debug=False, port=5110)


# retriever results example
#[Document(page_content='reaches  out  towards  more  Cosmic  truth,  his \ntendency  to  consolidate  and  formalise  his  pene-\ntrations  of  previous  Spiderisms  remains.  So  w,e \nfind that truths  which  pioneers  have  agonised  to \nattain  become  commonplace,  and  form  part  of \nthe  Spiderism  of  a  succeeding  generation.', metadata={'source': 'D:\\projets\\projets2023\\ia\\localGPT/SOURCE_DOCUMENTS\\fsr_1955_v_1_n_3.pdf'}),Document(page_content='Limited  Logic \nIf  I  could  speak  to  a  spider,  I  should  find  it \nquite  impossible  to  convey  to  its  limited  intelli-\ngence  any  conception  of  our  own  human  world. \nMy  obstacle  would  be  its  Spiderism.  Everything \nthat I said would be translated into terms related \nto  its  spider-habits  and  spider-conceptions.  It \nwould have its own limited " logic." Facts which \nfitted into that " logic " would be accepted. Facts \nwhich  did  not  would  be  rejected.  A  man\'s  face \nhas  no  significance  as  a  man\'s  face  to  a  spider. \n\n13 \n\n\x0cA  piece  of  newsprint is  probably  a  kind  of  leaf. \nSpiderism prevents the spider from attaining any \ndegree  of  Cosmic  truth.', metadata={'source': 'D:\\projets\\projets2023\\ia\\localGPT/SOURCE_DOCUMENTS\\fsr_1955_v_1_n_3.pdf'})]

