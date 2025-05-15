import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from typing import Literal

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Can not find GOOGLE_API_KEY in .env")

# Khởi tạo LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Prompt để phân loại đầu vào
route_prompt = ChatPromptTemplate.from_messages([
    ("system", "Route the query to either 'animal' or 'vegetable' expert."),
    ("human", "{input}")
])

class RouteQuery(TypedDict):
    destination: Literal["animal", "vegetable"]

# Create chain: prompt -> LLM -> structured output
chain = route_prompt | llm.with_structured_output(RouteQuery)

try:
    result = chain.invoke({"input": "What color are carrots?"})
    print(result["destination"])
except Exception as e:
    print(f"Failed: {e}")