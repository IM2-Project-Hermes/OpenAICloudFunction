from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
import functions_framework


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

    return chain({"question": question}, return_only_outputs=True)


if __name__ == '__main__':
    print(ask_llm("How does the onboarding process work?"))


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
    except:
        return {
            "status": 500,
            "error": "An error occurred"
        }
