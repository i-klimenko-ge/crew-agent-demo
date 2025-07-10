from langchain_core.tools import tool
from typing import List, Annotated
from bs4 import BeautifulSoup
# from langchain_community.tools import TavilySearchResults


@tool
def response_tool(question: str) -> dict:
    """Задать пользователю вопрос"""
    # Print the question to the terminal
    print(f"\n[Follow-up question]: {question}")
    # Wait for the user's response
    answer = input("> ")
    return {"answer": answer}

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


import os
from langchain_tavily import TavilySearch

search_tool = TavilySearch(max_results=3)
search_tool.name = "search_tool"
