import asyncio
import copy
import glob
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import cast, get_args

import boto3
import nest_asyncio
import streamlit as st
import yaml

from agent_cortex import run_cortex_agents
from agent_cortex import execute_sql as execute_snaowflake_sql

nest_asyncio.apply()

os.environ["DEV"] = "true"

async def write_cortex_result(response):
    text = response.get('text', '')
    sql = response.get('sql', '')
    citations = response.get('citations', '')
    results = response.get('results', '')
    
    # Add assistant response to chat
    if text:
        st.markdown(text.replace("•", "\n\n"))
        if citations:
            st.write("Citations:")
            for citation in citations:
                doc_id = citation.get("doc_id", "")
                doc_text = citation.get("doc_text", "") # 用于session_state中保存
                if doc_id and not doc_text:
                    query = f"SELECT transcript_text FROM sales_conversations WHERE conversation_id = '{doc_id}'"
                    doc_result = await execute_snaowflake_sql(query)
                    doc_result_df = doc_result.to_pandas()
                    if not doc_result_df.empty:
                        doc_text = doc_result_df.iloc[0, 0]
                    else:
                        doc_text = "No doc available"
                    citation['doc_text'] = doc_text
                if doc_text:
                    with st.expander(f"[{citation.get('source_id', '')}]"):
                        st.write(doc_text)

    # Display SQL if present
    if sql:
        st.markdown("### Generated SQL")
        st.code(sql, language="sql")
        if results:
            st.write("### SQL Result from Snowflake")
            st.dataframe(results.get('data'))

async def main():
    """
    Main function for the Streamlit application. Handles chat interface setup,
    model initialization, user input processing, and chat history management.

    Returns:
        None
    """
    st.title("客服分析助手")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if 'user' == message['role']:
            with st.chat_message(message['role']):
                st.write(message['content'])    
        else:
            with st.chat_message(message['role']):
                await write_cortex_result(message['content'])


    if query := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            # st.write_stream(streaming(agent_stream))
            response = await run_cortex_agents(query)
            await write_cortex_result(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
