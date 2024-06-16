from dotenv import load_dotenv

from auto_generate_prompt import execute_langgraph, init_agent_state, get_result_from_state
from langchain_community.llms import Tongyi

load_dotenv("default.env")

# user's request
request = f"""
- 找出下面文字中的时间
- 用JSON返回, 只有一个字段'Key'
"""
user_input = "如果从2021年初以来持续三年多的调整来看，A股基本是一个负回报的市场。但如果将时间拉长到2018年初，尽管经历了大幅波动，但持有好股票的投资者依然取得了不俗的成绩。"
expect_output = "[{'Key': '2021'},{'Key': '2018'}]"

test_cases = [{'user_input': '在2018年至今的6年时间里，A股仅有两年的幸福时光，却有四年惨淡的调整。上证指数2018年初为3400点，当前上证指数仅有3000点。', 'expect_output': "[{'Key': '2018'}]"}]


def get_target_model():
    return Tongyi()


agent_state = init_agent_state(
    request=request,
    user_input=user_input, 
    expect_output=expect_output,
    test_cases=test_cases,
    target_LLM=get_target_model())
result_agent_state = execute_langgraph(agent_state)
print(get_result_from_state(result_agent_state))