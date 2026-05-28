from pydantic import BaseModel
from typing import Annotated, List, Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, AIMessageChunk
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import InMemorySaver
from scout.tools import query_db, generate_visualization
from scout.prompts import prompts

class  ScoutState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = []
    chart_json: str = ""

llm = ChatOpenAI(name='Scout', model="gpt-4o-mini")
# llm.invoke('hi')

from langchain_core.tools import tool
@tool
def raise_number_to_the_power_of(a: float, b:float) -> str:
    """Raise the number a to the power of b"""
    return a**b

tools = [raise_number_to_the_power_of]
llm_w_tools = llm.bind_tools(tools)

def assistant_node(state: ScoutState) -> ScoutState:
    response = llm_w_tools.invoke(state.messages)
    state.messages.append(response)
    return state

#### Error
# def assistant_router(state: ScoutState) -> str:
#     last_message = state.messages[-1]
#     if "tool_calls" in last_message:
#         return "tools"
#     else:
#         END

def assistant_router(state: ScoutState) -> str:
    last_message = state.messages[-1]
    # Use attribute access and check if the list is not empty
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END # Ensure you return the END constant


# result = assistant_node(state)
# print(result.model_dump_json(indent=2))

builder = StateGraph(ScoutState)
builder.add_node(assistant_node)
builder.add_node(ToolNode(tools), "tools")
builder.add_edge(START, "assistant_node")
builder.add_conditional_edges(
    "assistant_node",
    assistant_router,
    ["tools", END]
)
builder.add_edge("tools", "assistant_node")
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

from IPython.display import display, Image
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

config = {"configurable": {"thread_id": "1"}}  # You can use any thread_id string or integer


######## conversations ##########
#1
state = ScoutState(
    messages=[HumanMessage(content="hey scout, what is 4.13 raised to the power of 8.16")],
    chart_json="this is chart json"
)

result = graph.invoke(input=state,
                      config=config)

#2
state = ScoutState(
    messages=[HumanMessage(content="What did you just raise the power by")],
    chart_json="this is chart json"
)

result = graph.invoke(input=state,
                      config=config,)





class Agent:
    """
    Agent class for implementing Langgraph agents.

    Attributes:
        name: The name of the agent.
        tools: The tools available to the agent.
        model: The model to use for the agent.
        system_prompt: The system prompt for the agent.
        temperature: The temperature for the agent.
    """
    def __init__(
            self, 
            name: str, 
            tools: List = [query_db, generate_visualization],
            model: str = "gpt-4.1-mini-2025-04-14", 
            system_prompt: str = "You are a helpful assistant.",
            temperature: float = 0.1
            ):
        self.name = name
        self.tools = tools
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature
            ).bind_tools(self.tools)
        
        self.runnable = self.build_graph()


    def build_graph(self):
        """
        Build the LangGraph application.
        """
        def scout_node(state: ScoutState) -> ScoutState:
            response = self.llm.invoke(
                [SystemMessage(content=self.system_prompt)] +
                state.messages
                )
            state.messages = state.messages + [response]
            return state
        
        def router(state: ScoutState) -> str:
            last_message = state.messages[-1]
            if not last_message.tool_calls:
                return END
            else:
                return "tools"

        builder = StateGraph(ScoutState)

        builder.add_node("chatbot", scout_node)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "chatbot")
        builder.add_conditional_edges("chatbot", router, ["tools", END])
        builder.add_edge("tools", "chatbot")

        return builder.compile(checkpointer=MemorySaver())
    

    def inspect_graph(self):
        """
        Visualize the graph using the mermaid.ink API.
        """
        from IPython.display import display, Image

        graph = self.build_graph()
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))


    def invoke(self, message: str, **kwargs) -> str:
        """Synchronously invoke the graph.

        Args:
            message: The user message.

        Returns:
            str: The LLM response.
        """
        result = self.runnable.invoke(
            input = {
                "messages": [HumanMessage(content=message)]
            },
            **kwargs
        )

        return result["messages"][-1].content
    

    def stream(self, message: str, **kwargs) -> Generator[str, None, None]:
        """Synchronously stream the results of the graph run.

        Args:
            message: The user message.

        Returns:
            str: The final LLM response or tool call response
        """
        for message_chunk, metadata in self.runnable.stream(
            input = {
                "messages": [HumanMessage(content=message)]
            },
            stream_mode="messages",
            **kwargs
        ):
            if isinstance(message_chunk, AIMessageChunk):
                if message_chunk.response_metadata:
                    finish_reason = message_chunk.response_metadata.get("finish_reason", "")
                    if finish_reason == "tool_calls":
                        yield "\n\n"

                if message_chunk.tool_call_chunks:
                    tool_chunk = message_chunk.tool_call_chunks[0]

                    tool_name = tool_chunk.get("name", "")
                    args = tool_chunk.get("args", "")

                    
                    if tool_name:
                        tool_call_str = f"\n\n< TOOL CALL: {tool_name} >\n\n"

                    if args:
                        tool_call_str = args
                    yield tool_call_str
                else:
                    yield message_chunk.content
                continue


# Define and instantiate the agent 
agent = Agent(
        name="Scout",
        system_prompt=prompts.scout_system_prompt
        )
graph = agent.build_graph()
