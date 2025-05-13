# This to show how embedding of multiple query is done
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embed_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32) 

documents = [
    "delhi is the capital of India",
    "Ranchi is the capital of Jharkhand",
    "Kolkata is the capital of West Bengal"
]

result = embed_model.embed_documents(documents)

print(str(result))
