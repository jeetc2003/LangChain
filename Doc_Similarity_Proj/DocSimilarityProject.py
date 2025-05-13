from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

load_dotenv()

embed_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.",
]

query = "tell me about bumrah"

doc_embeddings = embed_model.embed_documents(documents)  # 5 * 322 dim space
query_embedding = embed_model.embed_query(query)  # 322 dim space in 1d

scores = cosine_similarity([query_embedding], doc_embeddings)[
    0
]  # as cosine similarity requires 2D vectors as the 2 apparatus # also as the output is a 2d vector therefore we take [0]

# print(scores)

scores = [round(float(x), 3) for x in scores]

index, score = sorted(list(enumerate((scores))), key=lambda x: x[1], reverse=True)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)