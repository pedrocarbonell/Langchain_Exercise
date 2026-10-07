import asyncio
import os
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-6-luna",
    temperature=1,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("API_KEY"),
)
parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un asistente útil. Responde de forma clara y concisa."),
        ("human", "Pregunta: {pregunta}"),
    ]
)

chain = prompt_template | model | parser

async def main():
    resultado = await chain.ainvoke(
        {"pregunta": "Cual es la distancia de la tierra al sol"}
    )
    print(resultado)

if __name__ == "__main__":
    asyncio.run(main())








