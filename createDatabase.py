import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.text_splitter import CharacterTextSplitter, SentenceTransformersTokenTextSplitter
import uuid
from PyPDF2 import PdfReader
from dotenv import dotenv_values


# Read the file
def process_file(file_path):
    """
    Read the file and return the text
    :param: file_path: path to the file
    :return: text of the file
    """
    reader = PdfReader(file_path)
    length = len(reader.pages)
    text = ""
    for i in range(length):
        page = reader.pages[i]
        text += page.extract_text()

    return text


def find_name_from_source(file_path):
    """
    Extract the name of the file from the file path
    :param: file_path: path to the file
    :return: name of the file
    """
    filename = file_path.split("/")[-1]  # Extract the last part of the string after splitting by "/"
    name = filename.split(".")[0]  # Extract the part before the dot (file extension)

    return name


def create_collection_with_sample_data():
    """
    Create a new collection in the database
    :return:
    """

    # Get the database
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="db"
    ))

    # openAI embedding function
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=dotenv_values(".env")['OPENAI_API_KEY'],
        model_name="text-embedding-ada-002"
    )

    try:
        collection = chroma_client.create_collection(
            name="documents",
            embedding_function=embedding_function
        )
        print('Collection created')
    except:
        print('Collection already exists')
        return

    file_paths = [
        "data/Automotive_SPICE_for_Cybersecurity_EN.pdf",
        "data/Criteria_for_car-washes_conforming_to_VDA_specifications.pdf",
        "data/kvBAVQ-Automotive_SPICE_PAM_31_EN.pdf",
        "data/Minimizing_risks_in_the_supply_chain.pdf",
        "data/radar_handbook.pdf",
        "data/VDA_Band_7_QDX_2021_English.pdf",
        "data/VDA_QMC_Vol_7_-_QDX_Data_Exchange_Requirements_V2.0.pdf",
        "data/VDA_Volume_Assessment_of_Quality_Management_Methods__Guideline__1st_Edition__November_2017__Online-Document.pdf",
        "data/VDA_Volume_Quality-related_costs.pdf",
        "data/VDA_Volume_SU_OTA_1_edition_2020.pdf",
    ]

    # Initialize the text splitter
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000, chunk_overlap=100)

    for file_path in file_paths:

        # Split the text
        text = process_file(file_path)
        split_text = text_splitter.split_text(text)

        print(f'{len(split_text)} <- len(split_text)')

        # print(text)
        # print(f'${split_text} <- split_text')

        if len(split_text) == 0:
            print(f"Could not split {file_path}")
            continue

        sources = []
        ids = []

        # Add the metadata to each chunk
        for x in range(len(split_text)):
            # print(len(split_text[x]))
            # print(split_text[x])
            # print(len(split_text))

            collection.add(
                documents=[split_text[x]],
                metadatas=[{"name": find_name_from_source(file_path), "source": file_path}],
                ids=[str(uuid.uuid4()).replace("-", "")]
            )

        print(f"Saved {file_path}")


if __name__ == '__main__':
    create_collection_with_sample_data()
