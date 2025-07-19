import json
import time
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from state import AgentState
from tools import (
    response_tool,
    create_agent_tool,
    read_webpage_tool,
    current_date_tool,
    calculator_tool,
    send_email_tool,
    search_tool,
    write_note_tool,
    read_notes_tool,
)
from prompts import create_system_prompt, get_react_instructions

# ─── Utility Functions ─────────────────────────────────────────────────────────
def invoke_with_retry(model, messages, config, retries: int = 3, delay: float = 0.5):
    """Invoke the LLM with simple retry logic."""
    last_exception = None
    for _ in range(retries):
        try:
            return model.invoke(messages, config)
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            time.sleep(delay)
    raise last_exception

# Map name → tool
orchestrator_tools_by_name = {
    tool.name: tool
    for tool in [
        create_agent_tool,
        response_tool,
        write_note_tool,
        read_notes_tool,
    ]
}

secondary_tools_by_name = {
    tool.name: tool
    for tool in [
        read_webpage_tool,
        current_date_tool,
        calculator_tool,
        send_email_tool,
        search_tool,
        read_notes_tool,
    ]
}

def reflect_node(state: AgentState, config: RunnableConfig, model):
    """1) Reflect, plan & choose one tool call."""

    prompt = None

    addition_config = config.get("configurable", None)

    if addition_config:
        prompt = addition_config.get("prompt", None)

    if not prompt:
        prompt = create_system_prompt(list(secondary_tools_by_name.keys())) + get_react_instructions()

    system = SystemMessage(prompt)

    response = invoke_with_retry(
        model,
        [system] + list(state["messages"]),
        config,
    )

    return {"messages": [response]}

def use_tool_node(state: AgentState, tools_dict):
    """2) Execute the tool call chosen in reflect_node."""
    outputs = []
    last = state["messages"][-1]

    for call in last.tool_calls:
        result = tools_dict[call["name"]].invoke(call["args"])
        if call["name"] == 'question_user_tool':
            outputs.append(
                ToolMessage(
                    content="Выполнено обращение к пользователю.",
                    name=call["name"],
                    tool_call_id=call["id"],
                )
            )
            outputs.append(
                HumanMessage(
                    content=result["answer"]
                )
            )
        else:
            outputs.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    name=call["name"],
                    tool_call_id=call["id"],
                )
            )
    return {"messages": outputs}

def should_use_tool(state: AgentState):
    """If the last LLM output included a tool call, go to execute; otherwise end."""
    last = state["messages"][-1]
    return "use_tool" if last.tool_calls else "end"

def response_gotten(state: AgentState):
    """If the last LLM output included a tool call, go to execute; otherwise end."""
    last = state["messages"][-1]
    return "end" if last.name == "response_tool" else "reflect"