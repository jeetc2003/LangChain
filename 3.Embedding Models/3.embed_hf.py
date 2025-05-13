from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embed_model = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

result = embed_model.embed_query("Delhi is the captal of India")

print(str(result))