# coding: utf-8
"""Simple orchestrator for a multi agent system."""

from typing import List, Iterable

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool

from graph import get_graph
from prompts import create_system_prompt


class SubAgent:
    """Wrapper around a single-agent graph with custom prompt and tools."""

    def __init__(self, model, prompt: str, tools: List[BaseTool]):
        self.graph = get_graph(model, {t.name: t for t in tools})
        self.prompt = prompt

    def run(self, input_message: str) -> List[BaseMessage]:
        """Run the agent with the given human input."""
        state = {"messages": [HumanMessage(content=input_message)]}
        config = {"configurable": {"prompt": self.prompt}}
        stream = self.graph.stream(state, stream_mode="values", config=config)
        messages: List[BaseMessage] = []
        for step in stream:
            msg = step["messages"][-1]
            messages.append(msg)
        return messages


class Orchestrator:
    """Plan tasks and run sub agents sequentially."""

    def __init__(self, model):
        self.model = model

    def plan(self, user_task: str) -> List[str]:
        """Very naive planner that splits task by '.'"""
        steps = [s.strip() for s in user_task.split('.') if s.strip()]
        return steps or [user_task]

    def prompt_for_step(self, step: str) -> str:
        base = create_system_prompt()
        return f"{base}\n# Задача агента:\n{step}"

    def run(self, user_task: str, tools: List[BaseTool]) -> List[List[BaseMessage]]:
        steps = self.plan(user_task)
        results = []
        for step in steps:
            prompt = self.prompt_for_step(step)
            agent = SubAgent(self.model, prompt, tools)
            result = agent.run(step)
            results.append(result)
        return results
