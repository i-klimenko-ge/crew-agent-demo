import asyncio
import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from langchain_gigachat import GigaChat

from graph import get_graph
from tools import (
    response_tool,
    search_rag_tool,
    read_webpage_tool,
    current_date_tool,
    calculator_tool,
    send_email_tool,
    search_tool,
)

# Tool objects paired with human readable labels. This single source is used
# both for binding tools to the model and for populating the `/tools` endpoint
# so that the UI and backend stay in sync.
# Each tool is paired with a human readable label and a flag indicating whether
# the tool is required for the agent. Required tools are always selected in the
# UI and cannot be removed by the user.
TOOL_DEFS = [
    (response_tool, "уточнить у пользователя", False),
    (search_rag_tool, "поиск в документации", False),
    (search_tool, "поиск в интернете", False),
    (read_webpage_tool, "просмотр страниц", False),
    (current_date_tool, "текущая дата", False),
    (calculator_tool, "калькулятор", False),
    (send_email_tool, "отправить email", False),
]

# List of tools for the UI. Each entry contains the tool name (which must match
# the bound tool) and a user friendly label.
TOOLS = [
    {"name": tool.name, "label": label, "required": required}
    for tool, label, required in TOOL_DEFS
]

# Convenience list of names of required tools
REQUIRED_TOOL_NAMES = [tool.name for tool, _, required in TOOL_DEFS if required]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

with open("static/index.html", "r") as f:
    index_html = f.read()


def create_agent(tool_names=None, tools_by_name=None):
    api_key = os.getenv("GIGACHAT_API_KEY")
    model = GigaChat(
        credentials=api_key,
        scope="GIGACHAT_API_CORP",
        model="GigaChat-2-Max",
        base_url="https://gigachat-preview.devices.sberbank.ru/api/v1",
        verify_ssl_certs=False,
        profanity_check=False,
    )
    # Bind exactly the tools requested by the UI (or all of them by default).
    # Ensure required tools are always included.
    if tool_names is None:
        tool_names = [tool.name for tool, _, _ in TOOL_DEFS]
    else:
        # Guarantee presence of required tools
        for req in REQUIRED_TOOL_NAMES:
            if req not in tool_names:
                tool_names.append(req)
    tools_list = [tool for tool, _, _ in TOOL_DEFS if tool.name in tool_names]
    model = model.bind_tools(tools_list)

    if tools_by_name is None:
        from nodes import tools_by_name as base_tools

        tools_by_name = base_tools
    tools_by_name = {name: t for name, t in tools_by_name.items() if name in tool_names}

    return get_graph(model, tools_by_name=tools_by_name)


@app.get("/")
async def get_index():
    return HTMLResponse(index_html)


@app.get("/tools")
async def get_tools():
    return JSONResponse({"tools": TOOLS})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    loop = asyncio.get_running_loop()
    answer_queue: asyncio.Queue[str] = asyncio.Queue()
    waiting = {"status": False}

    class QuestionTool:
        name = "question_user_tool"
        description = "Ask user a follow-up question"

        def invoke(self, args):
            question = args if isinstance(args, str) else args.get("question", "")
            waiting["status"] = True
            asyncio.run_coroutine_threadsafe(
                websocket.send_text(json.dumps({"type": "question", "text": question})),
                loop,
            )
            answer = asyncio.run_coroutine_threadsafe(answer_queue.get(), loop).result()
            waiting["status"] = False
            return {"answer": answer}

    import nodes

    tools_dict = nodes.tools_by_name.copy()
    tools_dict["question_user_tool"] = QuestionTool()

    conversation = {"messages": []}
    config = {"configurable": {"prompt": None}}

    def run_graph(user_input: str, tool_names):
        local_graph = create_agent(tool_names=tool_names, tools_by_name=tools_dict)
        conversation["messages"].append(HumanMessage(content=user_input))
        stream = local_graph.stream(conversation, stream_mode="values", config=config)
        for step in stream:
            msg = step["messages"][-1]
            if msg in conversation["messages"]:
                continue
            conversation["messages"].append(msg)
            if getattr(msg, "name", "") == "response_tool":
                continue
            if isinstance(msg, AIMessage):
                for token in re.split(r"(\s+)", msg.content):
                    if token:
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text(token), loop
                        )
                        time.sleep(0.02)
                asyncio.run_coroutine_threadsafe(websocket.send_text("\n"), loop)
            else:
                asyncio.run_coroutine_threadsafe(
                    websocket.send_text(getattr(msg, "content", str(msg)) + "\n"),
                    loop,
                )

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            user_msg = payload.get("userMsg", "")
            tool_names = payload.get("tools", None)
            if tool_names is None:
                tool_names = [t[0].name for t in TOOL_DEFS]
            else:
                # Ensure required tools are present even if UI omits them
                for req in REQUIRED_TOOL_NAMES:
                    if req not in tool_names:
                        tool_names.append(req)

            system_prompt = payload.get("systemPrompt", "")
            extra_prompt = payload.get("extraPrompt", "")
            config["configurable"]["prompt"] = f"{system_prompt}{extra_prompt}"
            if waiting["status"]:
                await answer_queue.put(user_msg)
                continue
            await asyncio.to_thread(run_graph, user_msg, tool_names)
    except WebSocketDisconnect:
        pass
