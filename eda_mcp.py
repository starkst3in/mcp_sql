import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp_use import MCPAgent, MCPClient

async def main():
    # Load environment variables
    load_dotenv()

    # Define system prompt (constant for all queries)
    system_prompt = """
    You are a helpful data analysis assistant. Use only the provided data sources to process user inputs. Do not use external sources or your own knowledge base.
    """

    # Define question prompt with explicit reference to the MySQL database and table
    question_prompt = """
    Using the 'my_mysql' connection, analyze the data in the 'orders' table from the 'classicmodels' database. 
    Provide a summarized report including:
    - The key columns and their data types.
    - The total number of rows.
    - A brief summary of the data of the order date 2026 (status)
    """

    # Define path to zaturn_mcp.exe (not used in this config, but kept for reference)
    zaturn_path = os.path.expanduser("~/.local/bin/zaturn_mcp.exe")

    # Create configuration dictionary with dbutils server
    config = {
        "mcpServers": {
            "dbutils": {
                "command": "uvx",
                "args": [
                    "mcp-dbutils",
                    "--config",
                    r"C:\Users\akrde\Desktop\mcp\config.yaml"
                ]
            }
        }
    }

    try:
        # Create MCPClient from configuration dictionary
        client = MCPClient.from_dict(config)

        # Create LLM with Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )

        # Create agent with the client and system prompt
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=100,
            system_prompt=system_prompt
        )

        # Run the query with the question prompt
        result = await agent.run(
            question_prompt,
            max_steps=100,
        )
        print(f"\nResult: {result}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())