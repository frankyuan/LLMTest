import os
from typing import TypedDict, Callable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

MAX_TRY = 10 # Just try 10 times to generate prompt

class AgentState(TypedDict):
    request: str
    user_input: str
    expect_output: str
    test_cases: list
    current_prompt: str
    prompt_history: list
    best_prompt: str
    best_score: float
    actual_result: str
    current_score: float
    try_times: int
    target_LLM: Callable[[str], str]


def init_agent_state(request, user_input, expect_output, test_cases, target_LLM):
    agent_state = AgentState(
        request=request, 
        user_input=user_input, 
        expect_output=expect_output,
        test_cases=test_cases,
        current_prompt="",
        prompt_history=[],
        best_prompt="",
        best_score = 0,
        actual_result="",
        current_score = 0,
        try_times=0,
        target_LLM=target_LLM)
    return agent_state


def generate_prompt(agent_state):
    prompt = get_prompt_generation_prompt(agent_state)
    model = get_model_by_name()
    result = model.invoke(prompt)
    content = result.content
    agent_state["try_times"] += 1
    agent_state["current_prompt"] = content
    agent_state["prompt_history"].append(content)
    return agent_state


def evaluate_result_of_prompt(agent_state):
    target_LLM = agent_state["target_LLM"]
    test_cases = agent_state["test_cases"]
    current_prompt_template = agent_state["current_prompt"]
    passed_count = 0
    total_count = len(test_cases)
    for test_case in test_cases:
        current_prompt = current_prompt_template.replace("{user_input}", test_case["user_input"])
        result = target_LLM.invoke(current_prompt)
        prompt = get_evaluate_result_prompt(result, test_case["expect_output"])
        model = get_model_by_name() # TODO: maybe can use another LLM
        evaluate_result = model.invoke(prompt)
        evaluate_content = evaluate_result.content
        if 'True' in evaluate_content or 'true' in evaluate_content:
            passed_count += 1

    score = passed_count/total_count
    agent_state["current_score"] = score
    if score > agent_state["best_score"]:
        agent_state["best_score"] = score
        agent_state["best_prompt"] = current_prompt_template
    
    return agent_state


def can_finish_to_generate_prompt(agent_state):
    if agent_state["try_times"] >= MAX_TRY:
        return "true"
    
    if agent_state["current_score"] >= 0.9:
        return "true"
    
    return "false"


def execute_langgraph(agent_state):
    runnable = generate_graph()
    result = runnable.invoke(agent_state)
    return result


def generate_graph():
    graph = StateGraph(AgentState)
    graph.add_node("generate_prompt", generate_prompt)
    graph.add_node("evaluate_result_of_prompt", evaluate_result_of_prompt)

    graph.add_edge("generate_prompt", "evaluate_result_of_prompt")
    graph.add_conditional_edges("evaluate_result_of_prompt", can_finish_to_generate_prompt, {
        "true": END, "false": "generate_prompt"
    })

    graph.set_entry_point("generate_prompt")
    runnable = graph.compile()
    return runnable


def get_model_by_name(model_name="gpt-3.5-turbo"):
    model = ChatOpenAI(temperature=0, model_name=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"))
    return model


def get_result_from_state(agent_state):
    return agent_state["best_prompt"], agent_state["best_score"]


def get_prompt_generation_prompt(agent_state: AgentState):
    request = agent_state["request"]
    user_input = agent_state["user_input"]
    expect_output = agent_state["expect_output"]
    return f"""
    你是一个用来生成prompt的AI，可以根据`用户的需求`和`用户的输入`生成能够返回`预期输出`的prompt。
    `用户的输入`和`预期输出`都是用于帮助生成prompt的示例，和真正的输入和预期输出不同。

    返回给用户的prompt的逻辑要清楚。对于复杂的需求，可以使用COT的方式生成prompt。
    返回给用户的prompt 必须符合下面的规则：
    - 必须包含角色定义
    - 必须包含用户的输入。用户的输入用 {{user_input}} 这个格式表示，供后面的步骤用实际的数值替换。不能使用下面`用户的输入`，因为它只是用来帮助你生成prompt，不是用户实际的输入。
    - 不能包含预期的输出。因为预期的输出是根据prompt来生成的。
    - 直接返回生成的prompt

    #### 用户的需求:
    {request}
    ####

    #### 用户的输入:
    {user_input}
    ####

    #### 预期输出:
    {expect_output}
    ####

    在返回给用户前，检查是否达到下面的目标：
    - 生成的prompt符合每一条`用户的需求`
    - 生成的prompt必须能够返回`预期输出`
    - 生成的prompt必须包含类似`用户的输入`: {{user_input}}`的格式。
    - 直接返回生成的prompt，不需要返回任何其他信息，因为prompt会直接被作为后面的步骤给大模型的prompt。
    - prompt是文本，不是JSON对象，也不能包含代码。
    """


def get_evaluate_result_prompt(result, expect_output):
    return f"""
    你是一个用来评估结果的AI，可以检查`结果`是否符合`预期的输出`，并只能返回['True', 'False']数组中一个值。符合预期则返回True，不符合预期则返回False。

    结果:
    {result}

    预期的输出:
    {expect_output}
    """