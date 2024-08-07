import os
from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


# from llama_index.llms.groq import Groq
print("loading llm")
llm = Ollama(model="llama3.1:8b", request_timeout=10000.0)
print("loading embedder")
embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
print("setting llm and embedder")
Settings.llm = llm
Settings.embed_model = embed_model
print("loading data")
documents = SimpleDirectoryReader("data").load_data()
print("indexing data")
index = VectorStoreIndex.from_documents(documents, show_progress=True)
print("generating query engine")
query_engine = index.as_query_engine(similarity_top_k=3)
print("generating response")
response = query_engine.query("explain the scene where juliet dies ?")
print(response)