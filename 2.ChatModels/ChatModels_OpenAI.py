from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4", temperature=1, max_completion_tokens=10  # for example
)  

# Temperature is used for Randomness / Creative answers (Low-> 0-0.3)(High-> 0.7-1.5)
# max_completion_tokens = Max words in the answer


result = model.invoke("WHat is the capital of India")

print(result.content)
