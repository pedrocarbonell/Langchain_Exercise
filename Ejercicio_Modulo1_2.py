import asyncio
import os
import openai
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


# TODO 1: Define la clase Pydantic 'EntityExtraction'
# Debe tener: topic (str), entities (Lista de str), y sentiment_score (float entre 0 y 1)
class EntityExtraction(BaseModel):
    topic: str = Field(description="Tema principal del texto")
    entities: list[str] = Field(description="Lista de entidades encontradas")
    sentiment_score: float = Field(ge=0, le=1, description="Score de sentimiento entre 0 y 1.")


async def run_validated_chain(text: str):
    llm = ChatOpenAI(
        model="gpt-6-luna",
        reasoning_effort="medium",
        api_key=os.getenv("API_KEY"),
    )

    # TODO 2: Configura el modelo para usar la salida estructurada con Pydantic
    # Tip: Usa el método .with_structured_output()

    structured_llm = llm.with_structured_output(EntityExtraction)


    # TODO 3: Agrega una estrategia de reintento con .with_retry()
    # para que sea resiliente ante fallos de conexión (máximo 3 intentos).
    resilient_llm = structured_llm.with_retry(
        retry_if_exception_type=(openai.APIConnectionError, openai.RateLimitError),
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analiza el texto y extrae las entidades."),
        ("human", "{input}")
    ])

    # TODO 4: Une el prompt con el resilient_llm y ejecuta asíncronamente
    # No olvides manejar excepciones con try/except para capturar fallos de validación



    try:
        chain = prompt | resilient_llm
        response = await chain.ainvoke({"input": text})
        print(response)

    except ValidationError as e:
        print(f"La respuesta del LLM no cumple el esquema: {e.error_count()} errores")
        for err in e.errors():
            print(f" - {err['loc'][0]}: {err['msg']} (recibido: {err['input']!r})")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    sample_text = "Los paises de la OTAN son: "
    asyncio.run(run_validated_chain(sample_text))
