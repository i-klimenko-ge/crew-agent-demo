import time
import ipywidgets as widgets
from IPython.display import display, Markdown
from langchain_core.messages import HumanMessage, AIMessage

def launch_chat(graph, first_message=None, window_height=300, prompt = None):
    """
    Launch a widget‐based chat UI with a single Text input
    that will also serve as the follow-up prompt.
    """
    # --- shared state ---
    conversation = {"messages": []}
    expecting_followup = {"on": False}
    followup_result = {}
    config={"configurable": {"prompt": prompt}}

    # --- widgets ---
    output_area = widgets.Output(
        layout={'border': '1px solid #ccc',
                'height': f'{window_height}px',
                'overflow': 'auto'}
    )
    input_box = widgets.Text(
        placeholder='Type your message and hit Enter…',
        continuous_update=False
    )

    # --- the “tool” that asks follow-ups but uses the same input_box ---
    def question_user_tool(question: str) -> dict:
        # show the question in the output pane
        with output_area:
            display(Markdown(f"**Follow-up question:** {question}"))
        # tell the _on_submit handler that the NEXT enter-press is a follow-up
        expecting_followup["on"] = True
        followup_result.clear()
    
        # wait until user submits
        while "answer" not in followup_result:
            time.sleep(0.1)
    
        return {"answer": followup_result["answer"]}

    # --- send user → agent and stream responses ---
    def chat_with_agent(user_input: str):
        # append user
        conversation["messages"].append(HumanMessage(content=user_input))

        # stream agent reply
        stream = graph.stream(
            conversation,
            stream_mode="values",
            config=config
            )
        
        for step in stream:
            msg = step["messages"][-1]
            if msg in conversation["messages"]:
                continue
            if isinstance(msg, AIMessage):
                display(Markdown(f"**Agent:** {msg.content}"))
            else:
                # whatever else your graph might emit
                display(Markdown(f"```\n{msg.pretty_print()}\n```"))
            conversation["messages"].append(msg)

    # --- single handler for Enter key ---
    def _on_submit(change):
        if change['name'] != 'value': 
            return
        text = change['new'].strip()
        if not text:
            return
        input_box.value = ""  # clear immediately

        with output_area:
            if expecting_followup["on"]:
                # route this entry back to the follow-up tool
                followup_result["answer"] = text
                expecting_followup["on"] = False
            else:
                # normal chat flow
                if text.lower() in ('exit', 'quit'):
                    display(Markdown("*Goodbye!*"))
                    input_box.disabled = True
                else:
                    chat_with_agent(text)

    input_box.observe(_on_submit, names='value')

    # render
    display(widgets.VBox([output_area, input_box]))

    # if you need to inject question_user_tool into your graph,
    # you can monkeypatch it here before the first message:
    
    import nodes                         # adjust to your project path
    nodes.tools_by_name["question_user_tool"] = question_user_tool

    # optional seed message
    if first_message:
        with output_area:
            chat_with_agent(first_message)
