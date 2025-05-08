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
    You are an expert research assistant with access to Google Search. Your goal is to provide accurate, concise, and well-researched answers. Follow these guidelines:
    - Start by navigating to https://www.google.com to conduct your search.
    - Use the search box to search for relevant information.
    - Use reliable sources from Google Search results.
    - Summarize findings clearly and cite sources where applicable.
    - If information is ambiguous, state so and provide the best available data.
    - Maintain a neutral and professional tone.
    """

    # Define question prompt (can be changed for different queries)
    question_prompt = "Find the best restaurant in San Francisco USING GOOGLE SEARCH"

    # Create configuration dictionary
    config = {
        "mcpServers": {
            "playwright": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
                "env": {
                    "DISPLAY": ":1"
                }
            }
        }
    }

    # Create MCPClient from configuration dictionary
    client = MCPClient.from_dict(config)

    # Create LLM with Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7  # Optional: Adjust for creativity
    )

    # Create agent with the client and system prompt
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=30,
        system_prompt=system_prompt  # Pass constant system prompt
    )

    # Run the query with the question prompt
    result = await agent.run(
        question_prompt,
        max_steps=30,
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())