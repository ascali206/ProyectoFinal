import streamlit as st
from langchain_experimental.tools import PythonREPLTool
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
import datetime
import os
from typing import Any, Dict
from langchain_core.tools import Tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

load_dotenv()

def save_history(question, answer):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}: {question} -> {answer}\n")

def load_history():
    if os.path.exists("history.txt"):  # Corregido 'os.path.exist' a 'os.path.exists'
        with open("history.txt", "r") as f:
            return f.readlines()
    return []

def main():
    st.set_page_config(page_title="Agentes Interactivos de Python y CSVs",
                       page_icon="👾",
                       layout="wide")
    st.title("👾 Agente de Python Interactivo")
    st.markdown(
        """
        <style>
        .stApp { background-color: black; }
        .title { color: #ff4b4b; }
        .button { background-color: #ff4b4b; color: white; border-radius: 5px; }
        .input { border: 1px solid #ff4b4b; border-radius: 5px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    instrucciones = """
    - Siempre usa la herramienta, incluso si sabes la respuesta.
    - Debes usar código de Python para responder.
    - Eres un agente que puede escribir código.
    - Solo responde la pregunta escribiendo código, incluso si sabes la respuesta.
    - Si no sabes la respuesta escribe "No sé la respuesta".
    """


    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instrucciones=instrucciones)
    st.write("Prompt cargando...")

    tools = [PythonREPLTool()]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    python_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    python_agent_executor = AgentExecutor(
        agent=python_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    csv_agent_executor1: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="Aceros.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    csv_agent_executor2: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="Aleantes.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    csv_agent_executor3: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="Tratamientos.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    csv_agent_executor4: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="Presupuestocaja.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    def python_agent_executor_wrapper(original_prompt: str) -> Dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="CSV Agent 1",
            func=csv_agent_executor1.invoke,
            description="""Useful when you need to answer questions over Aceros.csv.
                            Takes as an input the entire question and returns the answer after running pandas calculations.""",
        ),
        Tool(
            name="CSV Agent 2",
            func=csv_agent_executor2.invoke,
            description="""Useful when you need to answer questions over Aleantes.csv
                            Takes as an input the entire question and returns the answer after running pandas calculations.""",
        ),
        Tool(
            name="CSV Agent 3",
            func=csv_agent_executor3.invoke,
            description="""Useful when you need to answer questions over Tratamientos.csv.
                                Takes as an input the entire question and returns the answer after running pandas calculations.""",
        ),
        Tool(
            name="CSV Agent 4",
            func=csv_agent_executor4.invoke,
            description="""Useful when you need to answer questions over Presupuestocaja.csv
                                Takes as an input the entire question and returns the answer after running pandas calculations.""",
        ),
        Tool(
            name="Python Agent",
            func=python_agent_executor.invoke,
            description="""Useful when you need to create a program or function to reach an objective, use python code""",
        ),

    ]
    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(
        agent=grand_agent,
        tools=tools,
        verbose=True
    )

    st.markdown("### Opciones: ")
    ejemplos = [
        "Calcula la suma de 2 y 3",
        "Genera una lista del 1 al 10",
        "Crea una función que calcule el factorial de un número"
    ]

    example = st.selectbox("Selecciona una opción:", ejemplos)

    if st.button("Ejecutar ejemplo"):
        user_input = example
        try:
            respuesta = python_agent_executor.invoke(
                input={"input": user_input, "agent_scratchpad": "", "instructions": instrucciones})
            st.markdown("### Respuesta del agente:")
            st.code(respuesta["output"], language="python")
            save_history(user_input, respuesta["output"])
        except ValueError as e:
            st.error(f"Error en el agente: {str(e)}")

    # Segundo título para el agente de archivos CSV
    st.title("📂 Agente de archivos CSVs o Agente de Python para programas")

    # Campo de entrada para preguntas sobre archivos CSV
    csv_question = st.text_input("Escribe tu pregunta sobre los archivos CSV o tu solicitud de programa:")

    # Botón para ejecutar preguntas relacionadas con archivos CSV
    if st.button("Procesar pregunta"):
        try:
            if "csv" in csv_question.lower():
                respuesta_csv = grand_agent_executor.invoke(input={"input": csv_question})
                print("csv \n")
                st.markdown("### Respuesta del agente sobre los CSVs:")
                st.write(respuesta_csv["output"])
                save_history(csv_question, respuesta_csv["output"])
            else :
                respuesta_csv = python_agent_executor.invoke(input={"input": csv_question, "agent_scratchpad": "", "instructions": instrucciones})
                st.markdown("### Respuesta del agente de python:")
                st.code(respuesta_csv["output"], language="python")
                save_history(csv_question, respuesta_csv["output"])
        except ValueError as e:
            st.error(f"Error en el agente: {str(e)}")

if __name__ == "__main__":
    main()
