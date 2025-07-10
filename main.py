from langchain_core.messages import HumanMessage, AIMessage
from colorama import init, Fore, Style, Back
from graph import get_graph
import os
from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from tools import provide_answer_tool, response_tool, search_rag_tool, read_webpage_tool, current_date_tool, calculator_tool, search_tool


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
            profanity_check=False
        )

tools_list = [
    provide_answer_tool, # Вернуть ответ пользователю
    response_tool,  # Задать вопрос пользователю
    search_rag_tool,     # Искать в документации SberDocs
    search_tool,         # Искать в интернете
    read_webpage_tool,   # Просмотреть содержимое страницы
    current_date_tool,   # Узнать текущую дату
    calculator_tool,     # Калькулятор
]

print("Tool names handed to graph:", [t.name for t in tools_list])

model = model.bind_tools(tools_list)

graph = get_graph(model)

prompt = None

conversation = {"messages": []}
config={"configurable": {"prompt": prompt}}

print("Чем могу помочь?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ('exit', 'quit'):
        print("Goodbye!")
        break

    first_human_message = HumanMessage(content=user_input)
    # Add the user's message as a HumanMessage
    conversation["messages"].append(first_human_message)

    # Stream through the agent
    stream = graph.stream(
        conversation,
        stream_mode="values",
        config=config
        )

    # Collect assistant messages
    for step in stream:
        msg = step["messages"][-1]
        try:
            # TODO: for first and last message. Maybe it shold made another way
            if msg in conversation["messages"]:
                continue
            if msg.name in ["provide_answer_tool", "question_user_tool"]:
                continue

            if isinstance(msg, AIMessage):
                print(f"{Fore.YELLOW}{msg.content}{Style.RESET_ALL}")
            else:
                msg.pretty_print()
            conversation["messages"].append(msg)
        except AttributeError:
            print(msg)
