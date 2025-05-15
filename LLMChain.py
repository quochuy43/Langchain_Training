import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence

# LEGACY
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Can not find GOOGLE_API_KEY in file .env")

prompt = ChatPromptTemplate.from_messages(
    [("user", "Tell me a {adjective} joke")],
)

chain = RunnableSequence(prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash"))

try:
    # Run chain with invoke
    result = chain.invoke({"adjective": "funny"})
    print(result.content)
except Exception as e:
    print(f"Failed in Gemini API: {str(e)}")
