import asyncio
from typing import Any, Dict
import requests
import logging, os

from strands import tool


from agent_gateway.tools import CortexAnalystTool, CortexSearchTool, PythonTool

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool

# Initialize FastMCP server
mcp = FastMCP("snowflake_strands_agent")

from dotenv import load_dotenv, find_dotenv
from snowflake.snowpark import Session

load_dotenv(find_dotenv())

connection_parameters = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
}

_session = Session.builder.configs(connection_parameters).getOrCreate()

# Enables Strands debug log level
logging.getLogger("strands").setLevel(logging.DEBUG)

# Sets the logging format and streams logs to stderr
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

SEMANTIC_MODEL_FILE = "customer_service_data.yaml"
SEMANTIC_MODEL_STAGE = "models"
CORTEX_SEARCH_CS_RECORDS_SEARCH = "CS_RECORDS_SEARCH" # 从Snow中根据 retrieval_columns 获取column时使用大些进行匹配
CORTEX_SEARCH_PRODUCT_REVIEWS_SEARCH = "PRODUCT_REVIEWS_SEARCH"

analyst_config = {
    "semantic_model": SEMANTIC_MODEL_FILE,
    "stage": SEMANTIC_MODEL_STAGE,
    "service_topic": "products info, orders, order logistic, product reviews, and customer service log",
    "data_description": "从snowflake数据库中搜索以下表: products, orders, reviews,  customer service log",
    "snowflake_connection": _session,
}
analyst = CortexAnalystTool(**analyst_config)


review_search_config = {
    "service_name": CORTEX_SEARCH_PRODUCT_REVIEWS_SEARCH,
    "service_topic": "product reviews in order",
    "data_description": "搜索订单产品的评论信息",
    "retrieval_columns": [
        "product_name", "brand", "category", "price", "total_amount",
        "rating", "review_content", "review_date"
    ],
    "snowflake_connection": _session,
}
review_search = CortexSearchTool(**review_search_config)

cs_log_search_config = {
    "service_name": CORTEX_SEARCH_CS_RECORDS_SEARCH,
    "service_topic": "customer service records/log",
    "data_description": "从订单产品的客服记录中进行搜索, 返回客服记录列表",
    "retrieval_columns": [
        "product_name", "brand", "category", "price", "total_amount",
        "resolution_status", "conversation_log", "created_date"
    ],
    "snowflake_connection": _session,
}
cs_log_search = CortexSearchTool(**cs_log_search_config)

@tool(description=analyst.description)
def analyst_tool(query):
    response = asyncio.run(analyst.query(query))
    print("response from analyst_tool:", response)
    return response

@tool(description=review_search.description)
def review_search_tool(query):
    response = asyncio.run(review_search.asearch(query))
    print("response from search_tool:", response)
    return response

"""
search_tool(query: str) -> list:
    - 从客服记录中进行搜索订单、产品相关的记录
    - 返回包含客服记录的列表
"""

@tool(description=cs_log_search.description)
def cs_log_search_tool(query):
    response = asyncio.run(cs_log_search.asearch(query))
    print("response from search_tool:", response)
    return response


mcp.add_tool(analyst.query, name="analyst_tool", description=analyst.description)
mcp.add_tool(review_search.asearch, name="review_search_tool", description=review_search.description)
mcp.add_tool(cs_log_search.asearch, name="cs_log_search_tool", description=cs_log_search.description)

snowflake_tools = [analyst_tool, review_search_tool, cs_log_search_tool]
# @mcp.tool()
# async def analyst_tool_mcp(query) -> Dict[str, Any]:
#     """
#     analyst_tool(query: str) -> str:
#         - takes a user's question about products, orders, logistic, reviews and customer services log, and query the data. Can only query from one table at one time.
#         - Returns the relevant data about requested
#     """
#     response = await analyst.query(query)
#     return response

# @mcp.tool()
# async def review_search_tool_mcp(query):
#     """
#     search_tool(query: str) -> list:
#         - Executes a search for relevant information about product reviews by customer
#         - Returns a list of relevant data from product reviews
#     """
#     response = await review_search.asearch(query)
#     return response

# @mcp.tool()
# async def cs_log_search_tool_mcp(query):
#     """
#     search_tool(query: str) -> list:
#         - 从客服记录中进行搜索订单、产品相关的记录
#         - 返回包含客服记录的列表
#     """
#     response = await cs_log_search.asearch(query)
#     return response

if __name__ == "__main__":
    mcp.run(transport='stdio')