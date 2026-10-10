# Composición de Cadenas: El Operador Pipe |

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()


# --- Sub-cadena 1: sentimiento ---
sentiment_chain = (
    ChatPromptTemplate.from_template(
        "Clasificá el sentimiento de esta reseña en una palabra "
        "(positivo/negativo/neutro): {review}"
    )
    | model
    | parser
)

# --- Sub-cadena 2: tema principal ---
topic_chain = (
    ChatPromptTemplate.from_template(
        "¿Cuál es el tema principal de esta reseña? Respondé en 3 palabras: {review}"
    )
    | model
    | parser
)

# --- Paso 1: corremos ambas sub-cadenas EN PARALELO ---
analysis = RunnableParallel(sentiment=sentiment_chain, topic=topic_chain)

# --- Paso 2: síntesis en una línea de ticket ---
ticket_prompt = ChatPromptTemplate.from_template(
    "Sentimiento: {sentiment}\nTema: {topic}\n"
    "Escribí una sola línea de ticket de soporte (máx. 15 palabras) para el equipo."
)

# --- Cadena completa: análisis paralelo -> síntesis ---
full_chain = analysis | ticket_prompt | model | parser

async def main():
    resultado = await full_chain.ainvoke(
        {"review": "Llevo 3 días esperando que respondan mi reclamo y nadie me contesta."}
    )
    print(resultado)

if __name__ == "__main__":
    asyncio.run(main())
