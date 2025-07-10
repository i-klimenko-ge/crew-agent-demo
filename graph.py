from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import reflect_node, use_tool_node, should_use_tool
from functools import partial

def get_graph(model, tools_by_name=None):
    if tools_by_name is None:
        from nodes import tools_by_name as default_tools
        tools_by_name = default_tools

    workflow = StateGraph(AgentState)

    reflect_with_tools = partial(reflect_node, model=model)
    use_tool_with_dict = partial(use_tool_node, tools_by_name=tools_by_name)

    # Step 1: reflect (plan & choose action)
    workflow.add_node("reflect", reflect_with_tools)
    # Step 2: execute (call the chosen tool)
    workflow.add_node("use_tool", use_tool_with_dict)

    # Start by reflecting
    workflow.set_entry_point("reflect")

    # If reflect_node emits a tool_call → go execute; else finish
    workflow.add_conditional_edges(
        "reflect",
        should_use_tool,
        {"use_tool": "use_tool", "end": END},
    )

    # After executing, loop back to planning
    workflow.add_edge("use_tool", "reflect")

    # Compile for use
    graph = workflow.compile()

    return graph

if __name__ == "__main__":
    import io
    from PIL import Image

    imageStream = io.BytesIO(graph.get_graph().draw_mermaid_png())
    imageFile = Image.open(imageStream)
    imageFile.save('graph.jpg')
