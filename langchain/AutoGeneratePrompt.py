from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

MAX_TRY = 10 # Just try 10 times to generate prompt

model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
graph = MessageGraph()

# Test if graph worked
# graph.add_node("oracle", model)
# graph.add_edge("oracle", END)
#
# graph.set_entry_point("oracle")
#
# runnable = graph.compile()
# result = runnable.invoke(HumanMessage(content="What is the meaning of life?"))
# print(result)

def generate_prompt():
    pass


def invoke_open_source_model(prompt):
    pass


def evaluate_result_of_prompt(prompt, result):
    pass


def can_finish_to_generate_prompt(state):
    pass


graph.add_node("generate_prompt", generate_prompt)
graph.add_node("invoke_open_source_model", invoke_open_source_model)
graph.add_node("evaluate_result_of_prompt", evaluate_result_of_prompt)
graph.add_conditional_edges("evaluate_result_of_prompt", can_finish_to_generate_prompt, {
    "true": END, "false": "generate_prompt"
})

graph.set_entry_point("generate_prompt")
runnable = graph.compile()
result = runnable.invoke()
print(result)