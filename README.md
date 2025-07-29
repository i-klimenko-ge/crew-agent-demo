# crew-agent-demo

This repository contains a simple multi-agent demo powered by LangGraph.
Agents communicate through a shared blackboard and can use a variety of tools
defined in `tools.py`.

## Available tools

- `response_tool` – send a message to the user.
- `read_webpage_tool` – download webpage text.
- `current_date_tool` – return the current date and day of week.
- `day_adder` – add a number of days to a date.
- `calculator_tool` – evaluate a math expression.
- `weather_tool` – fetch current weather via Open‑Meteo.
- `wikipedia_summary_tool` – get a short summary from Wikipedia.
- `plot_function_tool` – draw a plot of a math function.
- `send_email_tool` – send an email using Gmail SMTP.
- `write_note_tool` – add a note to the common board.
- `read_notes_tool` – read all notes.
- `search_tool` – perform a web search.
- `create_agent_tool` – spawn a secondary agent with its own tools.

The demo is intentionally minimal but showcases how multiple agents can
collaborate using these tools.
