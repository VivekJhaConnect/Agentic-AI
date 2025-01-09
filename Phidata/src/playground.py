from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
from phi.tools.yfinance import YFinanceTools
from phi.storage.agent.sqlite import SQLiteAgentStorage
from phi.playground import Playground, serve_playground_app


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

web_search_agent = Agent(
    name="web_search_agent",
    description="An agent that searches the web for information",
    model = Groq(id="llama-3.3-70b-versatile"),
    tool = [DuckDuckGo()],
    instructions="Search the web for information on the topic.",
    show_tool_calls=True,
    debug_mode=True,
)

# web_search_agent.print_response("What is the capital of France?", stream=True)

finance_agent = Agent(
    name="finance_agent",
    description="An agent that provides financial information",
    model = Groq(id="llama-3.3-70b-versatile"),
    tool = [YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Use table to display data",
    show_tool_calls=True,
    markdown=True,
)

# finance_agent.print_response("Summerize analyst recommendation of NVIDIA", stream=True)

app = Playground(agent=[finance_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
