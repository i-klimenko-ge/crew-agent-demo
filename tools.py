from langchain_core.tools import tool
from typing import List, Annotated
from prompts import add_subagent_reminder
from bs4 import BeautifulSoup
from blackboard import blackboard
from colorama import init, Fore, Style, Back
import urllib.parse

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

# ─── Add Days To Date ───
@tool
def day_adder(
    date: Annotated[str, "дата в формате YYYY-MM-DD"],
    days: Annotated[int, "количество дней"],
) -> dict:
    """Прибавляет к дате заданное количество дней."""
    try:
        dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:  # noqa: BLE001
        return {"error": str(exc)}
    new_date = dt + datetime.timedelta(days=int(days))
    return {"date": new_date.strftime("%Y-%m-%d")}

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
def weather_tool(city: Annotated[str, "название города"]) -> dict:
    """Возвращает текущую погоду из open-meteo.com."""
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "ru"},
    )
    geo_resp.raise_for_status()
    data = geo_resp.json()
    if not data.get("results"):
        return {"error": "city not found"}

    loc = data["results"][0]
    weather_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "current_weather": True,
        },
    )
    weather_resp.raise_for_status()
    w = weather_resp.json().get("current_weather", {})
    return {
        "city": loc.get("name"),
        "temperature": w.get("temperature"),
        "wind_speed": w.get("windspeed"),
    }


@tool
def send_email_tool(
    recipient: Annotated[str, "email получателя"],
    subject: Annotated[str, "тема письма"],
    body: Annotated[str, "текст письма"],
) -> dict:
    """Отправляет email через SMTP Gmail."""
    import os
    import smtplib
    from email.message import EmailMessage

    address = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("EMAIL_PASSWORD")

    if not address or not password:
        return {"error": "EMAIL_ADDRESS/EMAIL_PASSWORD not configured"}

    msg = EmailMessage()
    msg["From"] = address
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(address, password)
            smtp.send_message(msg)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

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


@tool
def wikipedia_summary_tool(query: Annotated[str, "поисковый запрос"]) -> dict:
    """Возвращает краткое введение из русской Википедии."""
    url = (
        "https://ru.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(query.replace(" ", "_"))
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return {"title": data.get("title"), "summary": data.get("extract")}


import numpy as np
import matplotlib.pyplot as plt


@tool
def plot_function_tool(
    expression: Annotated[str, "выражение от x"],
    start: Annotated[float, "начало диапазона"],
    end: Annotated[float, "конец диапазона"],
) -> dict:
    """Строит график функции и сохраняет PNG."""
    x = np.linspace(start, end, 200)
    try:
        y = eval(expression, {"__builtins__": None, "x": x, **np.__dict__})
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel(expression)
    plt.grid(True)
    file_name = "plot.png"
    plt.savefig(file_name)
    plt.close()
    return {"image": file_name}


import os
from langchain_tavily import TavilySearch

search_tool = TavilySearch(max_results=3)
search_tool.name = "search_tool"

# ─── Available tools for secondary agents ───
secondary_tools = [
    read_webpage_tool,
    current_date_tool,
    day_adder,
    calculator_tool,
    send_email_tool,
    weather_tool,
    wikipedia_summary_tool,
    plot_function_tool,
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
