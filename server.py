"""
Freshdesk MCP Server.

Exposes tools for querying Freshdesk Knowledge Base via RAG.
Uses LangChain's built-in integrations for embeddings, vector store, and RAG chain.

Stateless HTTP mode is enabled for serverless deployment (Vercel).
RAG resources load lazily on first tool call via `RAGAgent.ensure_initialized`.
"""
import logging
import os

# Force stateless mode for serverless environments (Vercel).
# FastMCP checks this env var to disable session tracking.
os.environ.setdefault("FASTMCP_STATELESS_HTTP", "true")

from fastmcp import FastMCP
from starlette.responses import JSONResponse

from rag import rag_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastMCP server
mcp = FastMCP("Freshdesk Knowledge Base")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    del request
    return JSONResponse({"status": "ok", "service": "freshdesk-mcp"})


@mcp.tool()
async def ask_freshdesk(question: str, session_id: str = "default_session") -> str:
    """
    Search the Knowledge Base for support articles, guides, and "How-to" information.

    WHEN TO USE:
    - Use this tool for questions about FEATURES, POLICIES, TROUBLESHOOTING, or SETUP GUIDES.
    - Examples: "How do I upgrade?", "What is the refund policy?", "Why is the wishlist button not showing?".
    - DO NOT use this for database lookups or querying specific user/merchant data (use `query_database` for that).

    Args:
        question: The user's natural language question.
        session_id: The current session ID (optional).

    Returns:
        Structured answer with citation links to the original articles.
    """
    try:
        logger.info("Received Freshdesk question | chars=%d", len(question or ""))

        # Resources are already initialized via lifespan — just query
        result = await rag_agent.answer_question(
            question=question,
            session_id=session_id
        )

        answer = result["answer"]
        source_text = result.get("formatted_citations", "")

        return f"{answer}{source_text}"

    except Exception as e:
        logger.exception("Error answering Freshdesk question")
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Provide context-specific error messages
        if "pinecone" in error_msg.lower() or "index" in error_msg.lower():
            return "⚠️ **Knowledge Base Unavailable**\n\nI couldn't connect to the Freshdesk knowledge base. The vector search service may be temporarily unavailable.\n\n**What you can do:**\n• Try again in a moment\n• Rephrase your question\n• If this persists, contact the team"
        elif "embedding" in error_msg.lower() or "model" in error_msg.lower():
            return "⚠️ **Search Processing Failed**\n\nI couldn't process your question for semantic search. This is likely a temporary issue.\n\n**What you can do:**\n• Try rephrasing your question\n• Try again in a moment"
        elif "timeout" in error_msg.lower():
            return "⚠️ **Search Timeout**\n\nThe Freshdesk search took too long to respond.\n\n**What you can do:**\n• Try a more specific question\n• Try again in a moment"
        else:
            return f"⚠️ **Freshdesk Search Failed**\n\nI couldn't retrieve an answer from the Freshdesk Knowledge Base ({error_type}).\n\n**What you can do:**\n• Try rephrasing your question\n• Try again in a moment\n• If this persists, contact the team"


@mcp.tool()
async def ask_freshdesk_with_llm(question: str, session_id: str = "default_session") -> str:
    """
    Search the Knowledge Base and get an LLM-generated answer with citations.
    
    This tool uses LangChain's LCEL RAG chain to synthesize answers from KB articles.
    
    WHEN TO USE:
    - Use when you want a conversational, synthesized answer (not just raw KB chunks).
    - Examples: "Explain how to set up loyalty rewards", "What are my payment options?".
    
    Args:
        question: The user's natural language question.
        session_id: The current session ID (optional).
    
    Returns:
        LLM-generated answer with citation links.
    """
    try:
        logger.info("Received Freshdesk LLM question | chars=%d", len(question or ""))
        
        result = await rag_agent.answer_question(
            question=question,
            session_id=session_id
        )
        
        answer = result["answer"]
        source_text = result.get("formatted_citations", "")
        
        return f"{answer}{source_text}"
        
    except Exception as e:
        logger.exception("Error answering Freshdesk question with LLM")
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Provide context-specific error messages
        if "pinecone" in error_msg.lower() or "index" in error_msg.lower():
            return "⚠️ **Knowledge Base Unavailable**\n\nI couldn't connect to the Freshdesk knowledge base. The vector search service may be temporarily unavailable.\n\n**What you can do:**\n• Try again in a moment\n• Rephrase your question\n• If this persists, contact the team"
        elif "embedding" in error_msg.lower() or "model" in error_msg.lower():
            return "⚠️ **Search Processing Failed**\n\nI couldn't process your question for semantic search. This is likely a temporary issue.\n\n**What you can do:**\n• Try rephrasing your question\n• Try again in a moment"
        elif "timeout" in error_msg.lower():
            return "⚠️ **Search Timeout**\n\nThe Freshdesk search took too long to respond.\n\n**What you can do:**\n• Try a more specific question\n• Try again in a moment"
        else:
            return f"⚠️ **Freshdesk Search Failed**\n\nI couldn't generate an answer from the Freshdesk Knowledge Base ({error_type}).\n\n**What you can do:**\n• Try rephrasing your question\n• Try again in a moment\n• If this persists, contact the team"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)


# ASGI app for serverless and production hosting (e.g., Vercel)
# stateless_http=True is required for serverless (no persistent sessions)
app = mcp.http_app(path="/mcp", transport="streamable-http", stateless_http=True)
