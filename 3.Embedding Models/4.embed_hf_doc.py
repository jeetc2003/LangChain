from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embed_model = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "delhi is the capital of India",
    "Ranchi is the capital of Jharkhand",
    "Kolkata is the capital of West Bengal"
]

result = embed_model.embed_documents(documents)

print(str(result))
