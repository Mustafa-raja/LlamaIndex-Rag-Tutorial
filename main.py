import os.path
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

print("loading llm")
llm_online = Groq(model="llama3.1-70b-8192", api_key="GROQ_API_KEY")
llm = Ollama(model="llama3.1:8b", request_timeout=10000.0)
print("loading embedder")
embed_model_online = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
print("setting up llm and embedder")
Settings.llm = llm_online
Settings.embed_model = embed_model_online
PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    print("loading data")
    documents = SimpleDirectoryReader("data").load_data()
    print("indexing data")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    print("Already indexed data, pulling vectors from storage")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
while True:
    question = input("Please enter your question or enter 'q' to exit: ")
    if(question.strip() == 'q'):
        exit()
    print("generating response")
    response = query_engine.query(question)
    print(response)
    
