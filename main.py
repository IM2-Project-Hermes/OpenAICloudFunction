import os
from langchain.llms import HuggingFacePipeline
import time
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.chains import RetrievalQAWithSourcesChain
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI


def ask_llm(question):
    load_dotenv()

    llm = OpenAI(
        temperature=0,
    )

    # all-MiniLM-L6-v2 embedding function from HuggingFace
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Initialize the database
    db = Chroma(
        collection_name="documents",
        persist_directory="db",
        embedding_function=hf,
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())

    # Receive Result
    result = chain({"question": question}, return_only_outputs=True)

    print(result)
    print("")

    # Output Result
    print("Answer: " + result["answer"].replace('\n', ' '))
    print("Source: " + result["sources"])

    return


if __name__ == '__main__':
    ask_llm("How does the onboarding process work?")
