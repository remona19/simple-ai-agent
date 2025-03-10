import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent, AgentType
from langchain.tools import TavilySearchResults

def format_news_response(news_results, llm):
    """Formats and summarizes news search results using Gemini."""
    
    # Extract relevant parts of the search results
    sources = [f"{item['title']}: {item['url']}" for item in news_results]
    news_text = "\n".join(sources)

    prompt = f"""
    Here are some recent news articles:\n{news_text}
    
    Please summarize the key points concisely in a human-readable format. Highlight important insights, 
    trends, or key takeaways. Format the output cleanly.
    """

    # Use Gemini to process and structure the news
    summary = llm.invoke([HumanMessage(content=prompt)])
    return summary.content  # Extract the text response

def main():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not google_api_key or not tavily_api_key:
        print("Error: Missing API keys. Set GOOGLE_API_KEY and TAVILY_API_KEY.")
        return

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)

    # Initialize Tavily Web Search Tool
    search_tool = TavilySearchResults(api_key=tavily_api_key)

    print("\n🤖 AI Chatbot with Web Search: Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if "news" in user_input.lower():
            print("Fetching the latest news...🔍")
            news_results = search_tool.invoke(user_input)

            if not news_results:
                print("No news found. Try another query.")
                continue

            structured_news = format_news_response(news_results, llm)
            print("\n📰 **Latest News Summary:**\n")
            print(structured_news)
        else:
            response = llm.invoke([HumanMessage(content=user_input)]).content
            print("AI:", response)

if __name__ == "__main__":
    main()

