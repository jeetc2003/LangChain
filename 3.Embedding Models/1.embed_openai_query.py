# Embedding Models -> Converts Text to Vector (n-dimension)
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embed_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32) 

result = embed_model.embed_query("Delhi is the captal of India")

print(str(result))
