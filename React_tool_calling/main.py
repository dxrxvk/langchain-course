from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# lang chain tool decorator 

@tool
def get_product_price(product:str) -> str:
    """ look up price in cataglogue, returns the price in USD"""
    print(f" >> Exectuting get_product_price(product='{product}')")
    prices = {"laptop": "999", "phone": "499", "tablet": "299"}
    return prices.get(product, "unknown product")

@tool
def apply_discount(price:float, discount_tier:str) -> float:
    """ apply discount to price
    available tiers: bronze,silver,gold"""
    print(f" >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentage = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discount_percentage.get(discount_tier,0)
    return round(price * (1 - discount / 100), 2)


# --agent loop--
@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "you have access to product catlog tool "
                "and a discount tool. \n\n"
                "Strict Rules - follow these exactly:\n"
                "1. NEVER guess or assume any product price."
                "You MUST call get_product_price first to get the real price. \n"
                "2. ONLY call apply_discount AFTER you have received"
                "a price from get_product_price. Pass the exact price"
                "returned by get_product_price - do NOT  pass a made-up number.\n"
                "3. NEVER  calculate discounts yourself using math."
                "Always use the apply_discount tool.\n"
                "4. If user does not specify a discount tier,"
                "ask them which tier to use - do NOT  assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range (1, MAX_ITERATIONS + 1):
        print (f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tools_calls = ai_message.tool_calls

        # if no tool calls, this is the final answer
        if not tools_calls:
            print(f"\nFinal Amswer: {ai_message.content}")
            return ai_message.content
        # process only the FIRST tool call - force one tool per iteration
        tool_call = tools_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict")

        observation  = tool_to_use.invoke(tool_args)
        print(f" [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Maximum iterations reached without a final answer.")

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")


