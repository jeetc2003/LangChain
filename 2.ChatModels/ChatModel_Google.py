from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")

review="Food was Good but did not like the Service ."

result = model.invoke("I have a restaurant and a customer gave this review - "+ review+". Write a short generous reply that we will surely improve and her review is very helpful for us to achieve perfectness. Only give the best text reply and no extra text for me to send it to the customer directly")

print(result.content)
