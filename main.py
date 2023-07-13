from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
import functions_framework
from chromadb.utils import embedding_functions
from dotenv import dotenv_values
from langchain.embeddings.openai import OpenAIEmbeddings


def ask_llm(question):
    load_dotenv()

    llm = OpenAI(
        temperature=0,
    )

    embedding_function = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=dotenv_values(".env")['OPENAI_API_KEY'],
    )

    # Initialize the database
    db = Chroma(
        collection_name="documents",
        persist_directory="db",
        embedding_function=embedding_function,
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())

    return chain({"question": question}, return_only_outputs=True)


if __name__ == '__main__':
    print(ask_llm(input("Enter a question: ")))


# Register an HTTP function with the Functions Framework
@functions_framework.http
def http_handler(request):
    if request.method != 'GET':
        return {
            "status": 400,
            "error": "Only GET requests are supported"
        }

    if request.args.get('question') is None:
        return {
            "status": 400,
            "error": "Missing required parameter: question"
        }

    question = request.args.get('question')

    try:
        result = ask_llm(question)
        return {
            "status": 200,
            "result": result
        }
    except Exception as e:
        return {
            "status": 500,
            "error": "An error occurred while processing the request",
            "error-message": str(e)
        }