from typing import TypedDict, Callable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

MAX_TRY = 10 # Just try 10 times to generate prompt

class AgentState(TypedDict):
    instruction: str
    data_source: list
    target_result: str
    current_prompt: str
    best_prompt: str
    best_score: float
    actual_result: str
    current_score: float
    try_times: int
    open_source_executor: Callable[[str], str]


def init_agent_state(instruction, data_source, target_result, open_source_executor=None):
    agent_state = AgentState(
        instruction=instruction, 
        data_source=data_source, 
        target_result=target_result,
        current_prompt="",
        best_prompt="",
        best_score = 0,
        actual_result="",
        current_score = 0,
        try_times=0,
        open_source_executor=open_source_executor)
    return agent_state


def generate_prompt(state):
    prompt = get_generate_prompt_prompt(state)
    model = get_model_by_name()
    result = model.invoke(HumanMessage(content=prompt))
    state.try_times += 1
    state.current_prompt = result  # TODO: need to extract prompt from result


def invoke_open_source_model(state):
    state.actual_result = state.open_source_executor(state.current_prompt)


def evaluate_result_of_prompt(state):
    prompt = get_evaluate_prompt_prompt(state)
    model = get_model_by_name()
    result = model.invoke(HumanMessage(content=prompt))
    state.current_score = result.score
    state.current_prompt = result.prompt
    if result.score > state.best_score:
        state.best_score = result.score
        state.best_prompt = state.current_prompt


def can_finish_to_generate_prompt(state):
    if state.try_times >= MAX_TRY:
        return "true"
    
    if state.current_score >= 0.9:
        return "true"
    
    return "false"


def execute_langgraph(agent_state):
    graph = MessageGraph()
    graph.add_node("generate_prompt", generate_prompt)
    graph.add_node("invoke_open_source_model", invoke_open_source_model)
    graph.add_node("evaluate_result_of_prompt", evaluate_result_of_prompt)

    graph.add_edge("generate_prompt", "invoke_open_source_model")
    graph.add_edge("invoke_open_source_model", "evaluate_result_of_prompt")
    graph.add_conditional_edges("evaluate_result_of_prompt", can_finish_to_generate_prompt, {
        "true": END, "false": "generate_prompt"
    })

    graph.set_entry_point("generate_prompt")
    runnable = graph.compile()
    result = runnable.invoke(agent_state)
    return result


def get_model_by_name(model_name="gpt-3.5-turbo"):
    model = ChatOpenAI(temperature=0, model_name=model_name)
    return model


def get_result_from_state(agent_state):
    return agent_state.best_prompt, agent_state.best_score


def get_generate_prompt_prompt(agent_state):
    pass


def get_evaluate_prompt_prompt(agent_state):
    pass