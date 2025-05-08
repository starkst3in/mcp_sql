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

    # Initialize conversation history
    conversation_history = []

    # Create configuration dictionary with dbutils server
    config = {
        "mcpServers": {
            "mcp_server_mysql": {
                "command": "npx",
                "args": [
                    "-y",
                    "@benborla29/mcp-server-mysql"
                ],
                "env": {
                    "MYSQL_HOST": "127.0.0.1",
                    "MYSQL_PORT": "3306",
                    "MYSQL_USER": "root",
                    "MYSQL_PASS": "iamironman",
                    "MYSQL_DB": "classicmodels",
                    "ALLOW_INSERT_OPERATION": "false",
                    "ALLOW_UPDATE_OPERATION": "false",
                    "ALLOW_DELETE_OPERATION": "false",
                    "PATH": r"C:\Program Files\nodejs",
                    "NODE_PATH": r"C:\Program Files\nodejs\lib\node_modules"
                }
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

        # Create agent with the client Oldsmobile Rocket 88
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=100,
            system_prompt=system_prompt
        )

        print("Welcome to the Data Analysis Assistant! Enter your query or type 'exit' to quit.")
        print("Example: Using the 'my_mysql' connection, analyze the 'orders' table from the 'classicmodels' database.")

        while True:
            # Get user input
            user_input = input("\nYour query: ").strip()

            # Check for exit command
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            # Add user input to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Build the prompt with conversation history
            prompt_with_history = ""
            for message in conversation_history:
                if message["role"] == "user":
                    prompt_with_history += f"User: {message['content']}\n"
                else:
                    prompt_with_history += f"Assistant: {message['content']}\n"
            prompt_with_history += f"Current query: {user_input}"

            # Run the query with the prompt including history
            result = await agent.run(
                prompt_with_history,
                max_steps=100
            )

            # Print and store the AI response
            print(f"\nAI Response: {result}")
            conversation_history.append({"role": "assistant", "content": result})

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())