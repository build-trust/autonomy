import os
import json
from typing import Optional

from autonomy import Agent, HttpServer, Model, Node, NodeDep, Tool
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dataclasses import asdict, is_dataclass
from enum import Enum

from linkup import LinkupClient


# Initialize Linkup client
linkup_client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))

# Maximum characters for search results to prevent context overflow
MAX_ANSWER_LENGTH = 1500
MAX_SNIPPET_LENGTH = 200


# === Linkup Tools ===

def linkup_search(
    query: str,
    depth: str = "standard",
    output_type: str = "sourcedAnswer",
    max_results: int = 3
) -> str:
    """
    Search the web using Linkup API for fresh, trusted information.

    IMPORTANT: Use this tool sparingly - make ONE comprehensive search rather than many small ones.

    Args:
        query: The search query string. Be specific and include all relevant context in a single query.
        depth: Search depth - always use "standard" (fast). Never use "deep".
        output_type: "sourcedAnswer" for natural language answer with citations (recommended).
        max_results: Maximum number of results (default 3, keep low for efficiency).

    Returns:
        JSON string with search results or sourced answer with citations.
    """
    try:
        kwargs = {
            "query": query,
            "depth": "standard",  # Force standard depth to avoid slow queries
            "output_type": output_type,
            "max_results": min(max_results, 3),  # Cap at 3 results
        }

        response = linkup_client.search(**kwargs)

        # Convert response to dict for JSON serialization
        if hasattr(response, 'answer'):
            # sourcedAnswer response - truncate answer if too long
            answer = response.answer
            if len(answer) > MAX_ANSWER_LENGTH:
                answer = answer[:MAX_ANSWER_LENGTH] + "..."

            # Limit sources and truncate snippets
            sources = []
            if hasattr(response, 'sources') and response.sources:
                for s in response.sources[:3]:  # Max 3 sources
                    snippet = getattr(s, 'snippet', '')
                    if len(snippet) > MAX_SNIPPET_LENGTH:
                        snippet = snippet[:MAX_SNIPPET_LENGTH] + "..."
                    sources.append({
                        "name": s.name[:100] if s.name else "",
                        "url": s.url,
                        "snippet": snippet
                    })

            result = {
                "answer": answer,
                "sources": sources
            }
        elif hasattr(response, 'results'):
            # searchResults response - limit and truncate
            results = []
            for r in response.results[:3]:  # Max 3 results
                content = getattr(r, 'content', getattr(r, 'snippet', ''))
                if len(content) > MAX_SNIPPET_LENGTH:
                    content = content[:MAX_SNIPPET_LENGTH] + "..."
                results.append({
                    "type": getattr(r, 'type', 'text'),
                    "name": getattr(r, 'name', '')[:100],
                    "url": getattr(r, 'url', ''),
                    "content": content
                })
            result = {"results": results}
        else:
            result = {"raw": str(response)[:500]}

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def linkup_fetch(url: str) -> str:
    """
    Fetch and extract content from a web page URL using Linkup API.
    Only use this if you absolutely need detailed content from a specific page.

    Args:
        url: The fully qualified URL to fetch (including http:// or https://).

    Returns:
        JSON string with the extracted markdown content from the page (truncated).
    """
    try:
        response = linkup_client.fetch(url=url)

        content = getattr(response, 'content', str(response))
        # Truncate content to prevent context overflow
        if len(content) > 2000:
            content = content[:2000] + "\n\n[Content truncated...]"

        result = {
            "url": url,
            "content": content
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "url": url})


# === FastAPI App ===

app = FastAPI(title="Signup Research API")


def json_serializer(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)


class ResearchRequest(BaseModel):
    name: str
    email: str
    organization: str


# Global agent reference
agent = None


AGENT_INSTRUCTIONS = """You are a research analyst who helps sales teams understand new signups.

When given a person's name, email, and organization, research them and provide a brief report.

## CRITICAL RULES - FOLLOW EXACTLY

1. **Make exactly 1 search call** - ONE comprehensive search combining company and person info
2. **Never use linkup_fetch** - the search results contain enough information
3. **Keep your response concise** - 2-3 sentences per section maximum

## Search Strategy

Make ONE search query combining everything:
- Example: "Acme Corp company products services [Person Name] role"

DO NOT make multiple searches. ONE search only.

## Output Format

After your single search, provide this brief report:

1. **Person Overview** - Their role (1 sentence, skip if not found)
2. **Company Overview** - What they do (2 sentences max)
3. **Products & Services** - Key offerings (2 sentences max)
4. **Recent News** - One notable item if found
5. **How Linkup Can Help** - One specific use case for their business:
   - Linkup is a web search API for AI applications
   - Examples: AI assistants, research automation, content generation, competitive intelligence

Total response should be under 300 words."""


@app.post("/api/research")
async def research(request: ResearchRequest, node: NodeDep):
    global agent

    # Initialize agent once and reuse
    if not agent:
        try:
            agent = await Agent.start(
                node=node,
                name="researcher",
                instructions=AGENT_INSTRUCTIONS,
                model=Model("claude-sonnet-4-5"),
                tools=[
                    Tool(linkup_search),
                    Tool(linkup_fetch),
                ],
            )
        except Exception as e:
            if "AlreadyExists" not in str(e):
                raise e

    # Construct the research message
    message = f"""Research this signup with exactly ONE search call:

- Name: {request.name}
- Email: {request.email}
- Organization: {request.organization}

Make one search, then write a brief report (under 300 words)."""

    async def stream_response():
        try:
            async for response in agent.send_stream(message, timeout=120):
                yield json.dumps(response.snippet, default=json_serializer) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream_response(), media_type="application/json")


# Serve static files (must be last)
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")


# Start the node with HTTP server
Node.start(http_server=HttpServer(app=app))
