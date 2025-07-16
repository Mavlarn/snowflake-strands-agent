import requests, os
from agent_gateway import Agent
from agent_gateway.tools import CortexAnalystTool, CortexSearchTool, PythonTool

from dotenv import load_dotenv
from snowflake.snowpark import Session
load_dotenv()
connection_parameters = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
}

_session = Session.builder.configs(connection_parameters).getOrCreate()

def html_crawl(url):
    response = requests.get(url)
    return response.text


SEMANTIC_MODEL_FILE = "customer_service_data.yaml"
SEMANTIC_MODEL_STAGE = "models"
CORTEX_SEARCH_CS_RECORDS_SEARCH = "CS_RECORDS_SEARCH" # 从Snow中根据 retrieval_columns 获取column时使用大些进行匹配
CORTEX_SEARCH_PRODUCT_REVIEWS_SEARCH = "PRODUCT_REVIEWS_SEARCH"

analyst_config = {
    "semantic_model": SEMANTIC_MODEL_FILE,
    "stage": SEMANTIC_MODEL_STAGE,
    "service_topic": "products info, orders, order logistic, product reviews, and customer service log",
    "data_description": "products, orders, reviews and  customer service data",
    "snowflake_connection": _session,
}

review_search_config = {
    "service_name": CORTEX_SEARCH_PRODUCT_REVIEWS_SEARCH,
    "service_topic": "product reviews in order",
    "data_description": "sales conversation call",
    "retrieval_columns": [
        "product_name", "brand", "category", "price", "total_amount",
        "rating", "review_content", "review_date"
    ],
    "snowflake_connection": _session,
}

cs_log_search_config = {
    "service_name": CORTEX_SEARCH_CS_RECORDS_SEARCH,
    "service_topic": "customer service records/log",
    "data_description": "customer service records about an order",
    "retrieval_columns": [
        "product_name", "brand", "category", "price", "total_amount",
        "resolution_status", "conversation_log", "created_date"
    ],
    "snowflake_connection": _session,
}

python_crawler_config = {
    "tool_description": "reads the html from a given URL or website",
    "output_description": "html of a webpage",
    "python_func": html_crawl,
}

# Tools Config
analyst = CortexAnalystTool(**analyst_config)
review_search = CortexSearchTool(**review_search_config)
cs_log_search = CortexSearchTool(**cs_log_search_config)
crawler = PythonTool(**python_crawler_config)

snowflake_tools = [ analyst, review_search, cs_log_search, crawler ]

orch_agent = Agent(
    snowflake_connection=_session,
    tools=snowflake_tools,
    planner_llm="claude-4-sonnet",
    agent_llm="claude-4-sonnet"
)
    