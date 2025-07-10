# crew-agent-demo

This project demonstrates a simple multi agent setup powered by LangGraph.
The entry point is `main.py` which runs an **orchestrator** agent. The orchestrator
creates a plan for the user request and then spawns a new agent for each step
of the plan. Each spawned agent gets its own prompt and list of tools.

Agents use the graph defined in `graph.py` and prompts from `prompts.yaml` as a
base. `orchestrator.py` provides lightweight logic for planning and sequential
execution of the step agents.
