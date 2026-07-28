#!/usr/bin/env python3
"""
OpenAI-Compatible MCP Client for SepsisEvalPipeline / GraphFlow MCP Server.

Allows executing MCP tools and running end-to-end pipeline automation using any OpenAI-compatible LLM endpoint
(e.g., OpenAI, Azure OpenAI, vLLM, Ollama, DeepSeek, Groq, OpenRouter, LocalAI).
"""

import os
import sys
import json
import argparse
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables from .env file
load_dotenv()



def convert_mcp_tools_to_openai(mcp_tools: List[Any]) -> List[Dict[str, Any]]:
    """Converts tools fetched from MCP server into OpenAI tool format."""
    openai_tools = []
    for tool in mcp_tools:
        # MCP tools schema matches JSON Schema object structure
        schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except Exception:
                schema = {"type": "object", "properties": {}}

        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema
            }
        })
    return openai_tools


async def run_mcp_client(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    mcp_script_path: str,
    max_turns: int = 10,
    verbose: bool = True
):
    """
    Main loop connecting to MCP server and processing tool calls via OpenAI client.
    """
    if verbose:
        print(f"Connecting to MCP Server at: {mcp_script_path}")
        print(f"Using OpenAI Endpoint: {base_url}")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}\n" + "=" * 60)

    # 1. Initialize OpenAI Client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 2. Configure MCP Stdio Server Parameters
    python_bin = sys.executable
    server_params = StdioServerParameters(
        command=python_bin,
        args=[mcp_script_path],
        env=dict(os.environ)
    )

    # 3. Connect to MCP Server via Stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Fetch tools from MCP Server
            tools_response = await session.list_tools()
            mcp_tools = tools_response.tools
            openai_tools = convert_mcp_tools_to_openai(mcp_tools)

            if verbose:
                print(f"[MCP Client] Loaded {len(openai_tools)} tools from MCP server:")
                for t in mcp_tools:
                    print(f"  - {t.name}: {t.description[:70]}...")
                print("=" * 60)

            # 4. Prepare Conversation History
            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI Assistant that manages and executes the SepsisEvalPipeline MCP server tools. "
                        "You can execute pipeline steps, inspect MLflow metrics, fetch G-Mean cutoffs, "
                        "run 1-hop spatial neighborhood inference, and explain patient predictions. "
                        "Use the provided MCP tools to complete the user's request accurately."
                    )
                },
                {"role": "user", "content": prompt}
            ]

            turn = 0
            while turn < max_turns:
                turn += 1
                if verbose:
                    print(f"\n[Turn {turn}] Sending request to LLM...")

                # Call OpenAI Chat Completion API
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                # Convert response message to dict format for message history
                assistant_msg = {
                    "role": "assistant",
                    "content": response_message.content
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

                if response_message.content and verbose:
                    print(f"[LLM Response]:\n{response_message.content}")

                # If no tool calls, LLM has completed its response
                if not tool_calls:
                    if verbose:
                        print("\n[MCP Client] Execution complete.")
                    return response_message.content

                # Process Tool Calls
                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        args = {}

                    if verbose:
                        print(f"\n[Tool Execution] Executing '{fn_name}' with args: {args}")

                    try:
                        # Invoke tool on MCP server
                        mcp_result = await session.call_tool(fn_name, args)
                        
                        # Extract content from MCP CallToolResult
                        output_content = []
                        for content in mcp_result.content:
                            if hasattr(content, "text"):
                                output_content.append(content.text)
                            else:
                                output_content.append(str(content))
                        
                        result_text = "\n".join(output_content)
                        if verbose:
                            print(f"[Tool Output ({fn_name})]:\n{result_text[:300]}..." if len(result_text) > 300 else f"[Tool Output ({fn_name})]:\n{result_text}")

                    except Exception as e:
                        result_text = json.dumps({"status": "error", "message": str(e)})
                        if verbose:
                            print(f"[Tool Error ({fn_name})]: {e}")

                    # Append tool result message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text
                    })

            print(f"[Warning] Reached max turns ({max_turns}). Stopping loop.")
            return messages[-1].get("content")


def main():
    parser = argparse.ArgumentParser(description="OpenAI-Compatible MCP Client for SepsisEvalPipeline")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "dummy-key"), help="API Key for OpenAI compatible endpoint")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="Base URL for OpenAI compatible endpoint")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"), help="Model identifier")
    parser.add_argument("--prompt", default="List all pipeline steps and check the status of the dashboard.", help="Prompt instruction for the LLM")
    parser.add_argument("--mcp-server", default=os.path.join(os.path.dirname(__file__), "mcp_server", "server.py"), help="Path to MCP server script")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum turn loops")

    args = parser.parse_args()

    asyncio.run(run_mcp_client(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        mcp_script_path=os.path.abspath(args.mcp-server if hasattr(args, 'mcp-server') else args.mcp_server),
        max_turns=args.max_turns
    ))


if __name__ == "__main__":
    main()
