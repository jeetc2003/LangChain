# Hugging Face has many open source api models ( most imp - LLama from MetaAI, deepseek ).
# We must use api from this hugging face website. 
# Mostly Text Data can be used

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint #to use api of hugging face

from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-Prover-V2-671B",
    max_new_tokens= 120,
    temperature=0.2,   
    repetition_penalty=1.1,
)

model = ChatHuggingFace(llm=llm)

review="Food was Good but did not like the Service ."

result = model.invoke("I have a restaurant and a customer gave this review - "+ review+". Don't write a letter but write a short generous reply that we will surely improve and her review is very helpful for us to achieve perfectness. Only give the best text reply and no extra text for me to send it to the customer directly. Don't write a letter and make the text short around 100 words with generous tone")

print(result.content)
