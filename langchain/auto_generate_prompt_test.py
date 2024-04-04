from auto_generate_prompt import AgentState, execute_langgraph, init_agent_state, get_result_from_state

# user's request
instruction = ""
data_source = [{}]
target_result = ""


def call_open_source_model(prompt):
    pass


agent_state = init_agent_state(
    instruction=instruction, 
    data_source=data_source, 
    target_result=target_result,
    open_source_executor=call_open_source_model)
result_agent_state = execute_langgraph(agent_state)
print(get_result_from_state(result_agent_state))