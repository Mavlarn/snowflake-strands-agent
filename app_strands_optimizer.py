import asyncio
import copy
import glob
import json
import os
import time
import ast
from pathlib import Path
from typing import cast, get_args

import boto3
import nest_asyncio
import streamlit as st
import yaml, logging
from strands import Agent
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from strands.types.content import ContentBlock, Message, Messages
from strands.types.media import ImageFormat
from strands_tools import current_time

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

nest_asyncio.apply()

os.environ["DEV"] = "true"

format = {"image": list(get_args(ImageFormat))}

builtin_tools = [current_time]

# Sets the logging format and streams logs to stderr
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

async def streaming(stream):
    """
    Asynchronous generator function that processes streaming data and generates text or tool usage information.

    Args:
        stream: Streaming response from the agent

    Yields:
        str: Formatted string of text data or tool usage information
    """
    async for event in stream:
        # If the event contains data, output it as text
        if "data" in event:
            # Output text
            data = event["data"]
            yield data
        # If the event contains a message, extract and output tool usage information
        elif "message" in event:
            # Output ToolUse
            message: Message = event["message"]
            # Extract tool usage information from message content
            for content in message["content"]:
                if "toolUse" in content.keys():
                    yield f"\n\n🔧 Using tool:\n```json\n{json.dumps(content, indent=2, ensure_ascii=False)}\n```\n\n"
                if "toolResult" in content.keys():
                    # tool_result_text = content['toolResult']['content'][0]['text']
                    # content['toolResult']['content'][0]['text'] = ast.literal_eval(tool_result_text)
                    yield f"\n\n🔧 Tool result:\n```json\n{json.dumps(content, indent=2, ensure_ascii=False)}\n```\n\n"


def convert_messages(messages: Messages, enable_cache: bool):
    """
    Function to add cache points to message history.

    Args:
        messages (Messages): Message history to convert
        enable_cache (bool): Flag indicating whether to enable caching

    Returns:
        Messages: Message history with cache points added
    """
    messages_with_cache_point: Messages = []
    user_turns_processed = 0

    for message in reversed(messages):
        m = copy.deepcopy(message)

        if enable_cache:
            if message["role"] == "user" and user_turns_processed < 2:
                if len([c for c in m["content"] if "text" in c]) > 0:
                    m["content"].append({"cachePoint": {"type": "default"}})  # type: ignore
                    user_turns_processed += 1
                else:
                    pass

        messages_with_cache_point.append(m)

    messages_with_cache_point.reverse()

    return messages_with_cache_point


async def main():
    """
    Main function for the Streamlit application. Handles chat interface setup,
    model initialization, user input processing, and chat history management.

    Returns:
        None
    """
    st.title("Snowflake SQL优化助手")

    with open("config/config.json", "r") as f:
        config = json.load(f)

    models = config["models"]
    bedrock_region = config["bedrock_region"]

    def select_chat(chat_history_file):
        st.session_state.chat_history_file = chat_history_file

    with st.sidebar:
        with st.expander(":gear: config", expanded=True):
            st.selectbox("LLM model", models.keys(), key="model_id")
            st.checkbox("Enable prompt cache", value=True, key="enable_prompt_cache")

            chat_history_dir = st.text_input(
                "chat_history_dir", value=config["chat_history_dir"]
            )

        st.button(
            "New Chat",
            on_click=select_chat,
            args=(f"{chat_history_dir}/{int(time.time())}.yaml",),
            use_container_width=True,
            type="primary",
        )

    if "chat_history_file" not in st.session_state:
        st.session_state["chat_history_file"] = (
            f"{chat_history_dir}/{int(time.time())}.yaml"
        )
    chat_history_file = st.session_state.chat_history_file

    if Path(chat_history_file).exists():
        with open(chat_history_file, mode="rt") as f:
            yaml_msg = yaml.safe_load(f)
            messages: Messages = yaml_msg
    else:
        messages: Messages = []

    for message in messages:
        for content in message["content"]:
            with st.chat_message(message["role"]):
                if "text" in content:
                    st.write(content["text"])
                elif "image" in content:
                    st.image(content["image"]["source"]["bytes"])

                if "toolUse" in content.keys():
                    st.write(f"\n\n🔧 Using tool:\n```json\n{json.dumps(content, indent=2, ensure_ascii=False)}\n```\n\n")
                if "toolResult" in content.keys():
                    st.write(f"\n\n🔧 Tool result:\n```json\n{json.dumps(content, indent=2, ensure_ascii=False)}\n```\n\n")

    enable_prompt_cache_system = False
    enable_prompt_cache_tools = False
    enable_prompt_cache_messages = False

    if st.session_state.enable_prompt_cache:
        cache_support = models[st.session_state.model_id]["cache_support"]
        enable_prompt_cache_system = True if "system" in cache_support else False
        enable_prompt_cache_tools = True if "tools" in cache_support else False
        enable_prompt_cache_messages = True if "messages" in cache_support else False

    image_support: bool = models[st.session_state.model_id]["image_support"]

    if prompt := st.chat_input(accept_file="multiple", file_type=format["image"]):
        with st.chat_message("user"):
            st.write(prompt.text)
            for file in prompt.files:
                if image_support:
                    st.image(file.getvalue())
                else:
                    st.warning(
                        "This model does not support images. Images will not be used."
                    )

        if prompt.files and image_support:
            image_content: list[ContentBlock] = []
            for file in prompt.files:
                if (file_format := file.type.split("/")[1]) in format["image"]:
                    image_content.append(
                        {
                            "image": {
                                "format": cast(ImageFormat, file_format),
                                "source": {"bytes": file.getvalue()},
                            }
                        }
                    )
            messages = messages + [
                {"role": "user", "content": image_content},
                {
                    "role": "assistant",
                    "content": [
                        {"text": "I will reference this media in my next response."}
                    ],
                },
            ]

        from agent_strands import snowflake_optimizer_tools

        
        if st.session_state.model_id == "deepseek-ai/DeepSeek-V3":
            SILICONFLOW_KEY = os.getenv("SILICONFLOW_KEY")
            the_model = OpenAIModel(
                client_args = {
                    "api_key": SILICONFLOW_KEY,
                    "base_url": "https://api.siliconflow.cn/v1"
                },
                model_id = "deepseek-ai/DeepSeek-V3",
            )
        else:
            the_model = BedrockModel(
                    model_id=st.session_state.model_id,
                    boto_session=boto3.Session(region_name=bedrock_region),
                    cache_prompt="default" if enable_prompt_cache_system else None,
                    cache_tools="default" if enable_prompt_cache_tools else None,
            )

        system_prompt = """
            You are a helpful assistant for analyzing and optimizing queries running on Snowflake to reduce resource consumption and improve performance.
            If the user's question is not related to query analysis or optimization, then politely refuse to answer it.

            Scope: Only analyze and optimize SELECT queries. Do not run any queries that mutate the data warehouse (e.g., CREATE, UPDATE, DELETE, DROP).

            YOU SHOULD FOLLOW THIS PLAN and seek approval from the user at every step before proceeding further:
            1. Identify Expensive Queries
                - For a given date range (default: last 7 days), identify the top 10 most expensive `SELECT` queries using the `SNOWFLAKE`.`ACCOUNT_USAGE`.`QUERY_HISTORY` view.
                - Criteria for "most expensive" can be based on execution time or data scanned.
            2. Analyze Query Structure
                - For each identified query, determine the tables being referenced in it and then get the schemas of these tables to under their structure.
            3. Suggest Optimizations
                - With the above context in mind, analyze the query logic to identify potential improvements.
                - Provide clear reasoning for each suggested optimization, specifying which metric (e.g., execution time, data scanned) the optimization aims to improve.
            4. Validate Improvements
                - Run the original and optimized queries to compare performance metrics.
                - Ensure the output data of the optimized query matches the original query to verify correctness.
                - Compare key metrics such as execution time and data scanned, using the query_id obtained from running the queries and the `SNOWFLAKE`.`ACCOUNT_USAGE`.`QUERY_HISTORY` view.
            5. Prepare Summary
                - Document the approach and methodology used for analyzing and optimizing the queries.
                - Summarize the results, including:
                    - Original vs. optimized query performance
                    - Metrics improved
                    - Any notable observations or recommendations for further action
            """

        def track_reasoning(**kwargs):
            if kwargs.get("reasoning") is True and "reasoning_content" in kwargs:
                print(f"REASONING: {kwargs['reasoningText']}")

        agent = Agent(
            model=the_model,
            system_prompt=system_prompt,
            # messages=convert_messages(messages, enable_cache=enable_prompt_cache_messages),
            messages=convert_messages(messages, enable_cache=False), # enable_cache=False for DeepSeek
            callback_handler=track_reasoning,
            tools=snowflake_optimizer_tools,
        )

        agent_stream = agent.stream_async(prompt=prompt.text)

        with st.chat_message("assistant"):
            st.write_stream(streaming(agent_stream))

        with open(chat_history_file, mode="wt") as f:
            yaml.safe_dump(agent.messages, f, allow_unicode=True)

    with st.sidebar:
        history_files = glob.glob(os.path.join(chat_history_dir, "*.yaml"))  # type: ignore

        for h in sorted(history_files, reverse=True)[:20]:  # latest 20
            st.button(h, on_click=select_chat, args=(h,), use_container_width=True)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
