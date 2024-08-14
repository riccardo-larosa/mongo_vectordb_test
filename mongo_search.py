import os, pprint
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate


def main():
    load_dotenv(override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    print(MONGODB_ATLAS_CLUSTER_URI)

    # Connect to your Atlas cluster
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

    # Define collection and index name
    db_name = "langchain_db"
    collection_name = "test"
    atlas_collection = client[db_name][collection_name]
    vector_search_index = "vector_index"

    # Load the PDF
    loader = PyPDFLoader("https://query.prod.cms.rt.microsoft.com/cms/api/am/binary/RE4HkJP")
    data = loader.load()

    # Split PDF into documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(data)

    # Print the first document
    print(docs[0])

    # Create the vector store
    #vector_store = MongoDBAtlasVectorSearch.from_documents(
    #    documents = docs,
    #    embedding = OpenAIEmbeddings(disallowed_special=()),
    #    collection = atlas_collection,
    #    index_name = vector_search_index
    #)

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI,
        db_name + "." + collection_name,
        OpenAIEmbeddings(disallowed_special=()) ,
        index_name = vector_search_index
    )

    if os.getenv("IMPORT_DATA") == "True":
        vector_search.add_documents(docs)

    query = "What were the compute requirements for training GPT 4"
    print(f"Searching for: {query}")
    #results = vector_store.similarity_search(query)
    results = vector_search.similarity_search(query)
    results = vector_search.similarity_search_with_score(
        query, 
        num_results=5, 
        score_threshold=0.7
    )
    #pprint.pprint(results)
    # Display results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()