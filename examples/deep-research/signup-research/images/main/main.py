import os
import json
import secrets
from datetime import datetime
from typing import Optional

from autonomy import Agent, FilesystemTools, HttpServer, Model, Node, NodeDep, Tool
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dataclasses import asdict, is_dataclass
from enum import Enum

from linkup import LinkupClient


# Initialize Linkup client
linkup_client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))

# Relaxed limits for deep research - allow much more content
MAX_ANSWER_LENGTH = 10000
MAX_SNIPPET_LENGTH = 2000


# === Linkup Tools ===

def linkup_search(
    query: str,
    output_type: str = "sourcedAnswer",
    max_results: int = 5
) -> str:
    """
    Search the web using Linkup API for fresh, trusted information.

    Use this tool multiple times for thorough research - one search per topic/step.

    Args:
        query: The search query string. Be specific and focused on one aspect per search.
        output_type: "sourcedAnswer" for natural language answer with citations (recommended).
        max_results: Maximum number of results (default 5).

    Returns:
        JSON string with search results or sourced answer with citations.
    """
    try:
        kwargs = {
            "query": query,
            "depth": "standard",  # Always use standard - deep causes timeouts and errors
            "output_type": output_type,
            "max_results": max_results,
        }

        response = linkup_client.search(**kwargs)

        # Convert response to dict for JSON serialization
        if hasattr(response, 'answer'):
            # sourcedAnswer response
            answer = response.answer
            if len(answer) > MAX_ANSWER_LENGTH:
                answer = answer[:MAX_ANSWER_LENGTH] + "..."

            sources = []
            if hasattr(response, 'sources') and response.sources:
                for s in response.sources[:5]:
                    snippet = getattr(s, 'snippet', '')
                    if len(snippet) > MAX_SNIPPET_LENGTH:
                        snippet = snippet[:MAX_SNIPPET_LENGTH] + "..."
                    sources.append({
                        "name": s.name[:200] if s.name else "",
                        "url": s.url,
                        "snippet": snippet
                    })

            result = {
                "answer": answer,
                "sources": sources
            }
        elif hasattr(response, 'results'):
            # searchResults response
            results = []
            for r in response.results[:5]:
                content = getattr(r, 'content', getattr(r, 'snippet', ''))
                if len(content) > MAX_SNIPPET_LENGTH:
                    content = content[:MAX_SNIPPET_LENGTH] + "..."
                results.append({
                    "type": getattr(r, 'type', 'text'),
                    "name": getattr(r, 'name', '')[:200],
                    "url": getattr(r, 'url', ''),
                    "content": content
                })
            result = {"results": results}
        else:
            result = {"raw": str(response)[:2000]}

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def linkup_fetch(url: str) -> str:
    """
    Fetch and extract content from a web page URL using Linkup API.
    Use this to get detailed content from specific pages like company websites or LinkedIn profiles.

    Args:
        url: The fully qualified URL to fetch (including http:// or https://).

    Returns:
        JSON string with the extracted markdown content from the page.
    """
    try:
        response = linkup_client.fetch(url=url)

        content = getattr(response, 'content', str(response))
        # Allow longer content for deep research
        if len(content) > 8000:
            content = content[:8000] + "\n\n[Content truncated...]"

        result = {
            "url": url,
            "content": content
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "url": url})


# === FastAPI App ===

app = FastAPI(title="Deep Research API")


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
    business_description: str





AGENT_INSTRUCTIONS_TEMPLATE = """You are a research analyst who helps sales teams understand new signups.

When given a person's name, email, and organization, research them and provide a detailed report.

## YOUR CLIENT'S BUSINESS

You are researching on behalf of a business with the following description:
{business_description}

## RESEARCH PROCESS

1. Search for company info (1-2 searches)
2. Search for the person's role and LinkedIn (1-2 searches)
3. Write the final report DIRECTLY as your response

## CRITICAL INSTRUCTION

After completing your searches, you MUST write out the complete research report as plain text in your response. Do NOT just save to files - the user needs to see the report directly in the chat.

## REPORT FORMAT

Write this report directly in your response:

# Research Report: [Person Name] at [Company]

## Executive Summary
2-3 sentence overview of who this person is and what their company does.

## Company Profile
- **Company**: Name and website
- **Industry**: What they do
- **Products/Services**: Key offerings
- **Size/Funding**: If known

## Person Profile
- **Name**: Full name
- **Role**: Current position
- **Background**: Professional experience
- **LinkedIn**: URL if found

## Recommendations for Outreach
- How the client's business (described above) could help them
- Specific use cases based on their industry
- Personalization hooks from research

## Sources
- List URLs consulted

IMPORTANT: Write the entire report as text output. Do not just save to files."""


@app.post("/api/research")
async def research(request: ResearchRequest, node: NodeDep):
    # Create timestamp for this research session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create unique agent name for this request to avoid AlreadyExists issues
    agent_name = f"researcher_{secrets.token_hex(4)}"

    # Format instructions with the business description
    instructions = AGENT_INSTRUCTIONS_TEMPLATE.format(
        business_description=request.business_description
    )

    # Initialize a fresh agent for this request
    agent = await Agent.start(
        node=node,
        name=agent_name,
        instructions=instructions,
        model=Model("claude-sonnet-4-5"),
        tools=[
            Tool(linkup_search),
            Tool(linkup_fetch),
            FilesystemTools(visibility="conversation"),
        ],
    )

    # Construct the research message
    message = f"""Research this new signup and write a report:

- **Name**: {request.name}
- **Email**: {request.email}
- **Organization**: {request.organization}

Do 2-4 web searches to gather information, then write the complete research report directly in your response.

CRITICAL: After your searches, write out the full formatted report as text. The user is reading your response directly - they cannot see files you save. Your text output IS the deliverable."""

    async def stream_response():
        try:
            async for response in agent.send_stream(message, timeout=300):
                yield json.dumps(response.snippet, default=json_serializer) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            # Clean up agent after request completes
            try:
                await Agent.stop(node, agent_name)
            except Exception:
                pass  # Ignore cleanup errors

    return StreamingResponse(stream_response(), media_type="application/json")


# Serve static files (must be last)
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")


# Start the node with HTTP server
Node.start(http_server=HttpServer(app=app))
