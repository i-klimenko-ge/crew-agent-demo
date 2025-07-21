from langchain_core.tools import tool
from typing import List, Annotated
from prompts import add_subagent_reminder
from bs4 import BeautifulSoup
from blackboard import blackboard
from colorama import init, Fore, Style, Back


@tool
def response_tool(message: str) -> dict:
    """Коммуникация с пользователем. Можно задать вопрос или предоставить развернутый ответ"""
    # Print the question to the terminal
    print(f"\n[Agent message]: {message}")
    blackboard.post("assistant", message)
    # Wait for the user's response
    answer = input("> ")
    blackboard.post("user", answer)
    return {"answer": answer}

@tool
def response_tool(response: Annotated[str, "сообщение для пользователя"]) -> dict:
    """Отправить сообщение пользователю. Это может быть вопрос или полный и развернутый ответ, либо вежливое прощание"""
    return {"answer": response}

import requests

# ─── Read & Extract Webpage Text ───
@tool
def read_webpage_tool(url: Annotated[str, "URL страницы"]) -> dict:
    """Парсит страницу и возвращает только текст (без разметки)."""
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return {"text": text}

import datetime

# ─── Current Date ───
@tool
def current_date_tool() -> dict:
    """Возвращает текущую дату в формате ГГГГ-ММ-ДД и день недели (на англ.)."""
    now = datetime.datetime.now()
    return {
        "date":       now.strftime("%Y-%m-%d"),
        "day_of_week": now.strftime("%A")
    }

import math

# ─── Current Date ───
@tool
def day_adder() -> dict:
    """Прибавляет к дате заданное количество дней"""
    pass

import math

# ─── Calculator ───
@tool
def calculator_tool(expression: Annotated[str, "Выражение для вычисления"]) -> dict:
    """Вычисляет математическое выражение и возвращает результат."""
    try:
        # безопасный eval с доступом только к math.*
        result = eval(expression, {"__builtins__": None}, math.__dict__)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def send_email_tool(
    recipient: Annotated[str, "email получателя"],
    subject: Annotated[str, "тема письма"],
    body: Annotated[str, "текст письма"],
) -> dict:
    """Отправляет email через SMTP Gmail."""
    return {"status": "sent"}


@tool
def write_note_tool(
    author: Annotated[str, "Имя агента"],
    content: Annotated[str, "Текст записи"],
) -> dict:
    """Добавляет запись на общую доску."""
    blackboard.post(author, content)
    return {"status": "written"}


@tool
def read_notes_tool() -> dict:
    """Возвращает все записи с доски."""
    return {"notes": blackboard.read()}


import os
from langchain_tavily import TavilySearch

search_tool = TavilySearch(max_results=3)
search_tool.name = "search_tool"

# ─── Available tools for secondary agents ───
secondary_tools = [
    read_webpage_tool,
    current_date_tool,
    calculator_tool,
    send_email_tool,
    search_tool,
    read_notes_tool,
]

secondary_tools_by_name = {tool.name: tool for tool in secondary_tools}


@tool
def create_agent_tool(
    name: Annotated[str, "Имя нового агента"],
    system_prompt: Annotated[str, "Системный промпт"],
    tools: Annotated[List[str], "Список инструментов"],
    message: Annotated[str, "Запрос для агента"],
) -> dict:
    """Создает вспомогательного агента и возвращает его ответ.

    Независимо от переданных инструментов, агент всегда получает возможность
    читать заметки через ``read_notes_tool``. Его итоговый ответ автоматически
    записывается в общие заметки.
    """
    from model import get_model
    from graph import get_graph

    # Always provide the reading note tool to the secondary agent
    tools = list(dict.fromkeys(tools + ["read_notes_tool"]))

    selected = [secondary_tools_by_name[t] for t in tools if t in secondary_tools_by_name]

    model = get_model(selected)
    graph = get_graph(model, tools_by_name={tool.name: tool for tool in selected})

    from langchain_core.messages import HumanMessage

    conversation = {"messages": [HumanMessage(content=message)]}
    system_prompt = add_subagent_reminder(system_prompt)
    config = {"configurable": {"prompt": system_prompt}}
    for step in graph.stream(conversation, stream_mode="values", config=config):
        msg = step["messages"][-1]
        print(f"{Fore.CYAN}{msg.content}{Style.RESET_ALL}")
        conversation = step

    final_msg = conversation["messages"][-1]
    result = getattr(final_msg, "content", str(final_msg))

    # store final result as a note for future agents
    blackboard.post(name, result)

    return {"result": result}
