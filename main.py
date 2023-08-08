from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import weaviate
from langchain.vectorstores import Weaviate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Load PDF documents
loader = DirectoryLoader('./docs', glob="**/*.pdf")
data = loader.load()

if not data:
    print("No documents loaded from directory.")
    exit()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

if not docs:
    print("Failed to split documents.")
    exit()

# Get embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Connect to Weaviate Cluster
auth_config = weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_KEY"))
WEAVIATE_URL = os.getenv("WEAVIATE_CLUSTER")

client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    auth_client_secret=auth_config
)

# Define schema
client.schema.delete_all() 
client.schema.get()

schema = {
    "classes": [
        {
            "class": "Chatbot",
            "description": "Documents for chatbot",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {"text2vec-openai": {"model": "ada", "type": "text"}},
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
            ],
        },
    ]
}

client.schema.create(schema)

vectorstore = Weaviate(client, "Chatbot", "content", attributes=["source"])

# Load text into the vectorstore
text_meta_pair = [(doc.page_content, doc.metadata) for doc in docs]

if not text_meta_pair:
    print("No document content and metadata pairs found.")
    exit()

texts, meta = list(zip(*text_meta_pair))
vectorstore.add_texts(texts, meta)

query = input("Enter your Query:")

# Retrieve text related to the query
docs = vectorstore.similarity_search(query, top_k=20)

if not docs:
    print("No similar documents found for the query.")
    exit()

# Define chain
chain = load_qa_chain(
    OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0), 
    chain_type="stuff"
)

# Create answer
response = chain.run(input_documents=docs, question=query)

if response:
    print(response)
else:
    print("Could not generate a response for the given query.")
