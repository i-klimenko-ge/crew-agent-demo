from colorama import Fore, Style
from langchain_core.messages import AIMessage, BaseMessage
from langchain_gigachat import GigaChat

import os

from orchestrator import Orchestrator
from tools import (
    provide_answer_tool,
    response_tool,
    search_rag_tool,
    read_webpage_tool,
    current_date_tool,
    calculator_tool,
    search_tool,
)


# Получаем ключ
api_key = os.getenv("GIGACHAT_API_KEY")
if not api_key:
    print("Error: GIGACHAT_API_KEY not found in environment variables")

# Инициализируем модель
model = GigaChat(
    credentials=api_key,
    scope="GIGACHAT_API_CORP",
    model="GigaChat-2-Max",
    base_url="https://gigachat-preview.devices.sberbank.ru/api/v1",
    verify_ssl_certs=False,
    profanity_check=False,
)

all_tools = [
    provide_answer_tool,
    response_tool,
    search_rag_tool,
    search_tool,
    read_webpage_tool,
    current_date_tool,
    calculator_tool,
]

model = model.bind_tools(all_tools)

print("Чем могу помочь?")

orchestrator = Orchestrator(model)

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    results = orchestrator.run(user_input, all_tools)

    for step_messages in results:
        for msg in step_messages:
            if isinstance(msg, AIMessage):
                print(f"{Fore.YELLOW}{msg.content}{Style.RESET_ALL}")
            elif isinstance(msg, BaseMessage):
                try:
                    msg.pretty_print()
                except Exception:
                    print(msg)
