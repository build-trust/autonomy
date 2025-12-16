from autonomy import Agent, HttpServer, Model, Node, NodeDep
from fastapi import FastAPI
import re

app = FastAPI()

# Cache of agents by model name
agents = {}

@app.post("/test")
async def test_model(request: dict, node: NodeDep):
    """
    Test endpoint that accepts a model name and message.

    Request body:
    {
        "model": "gpt-4o-mini",  # or "claude-sonnet-4-5", "claude-opus-4", etc.
        "message": "Say hello in exactly 3 words"
    }
    """
    model_name = request.get("model", "gpt-4o-mini")
    message = request.get("message", "Say hello")

    # Sanitize model name for use as agent name (only alphanumeric, hyphens, underscores)
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)

    # Get or create agent for this model
    if model_name not in agents:
        agents[model_name] = await Agent.start(
            node=node,
            name=f"tester_{safe_name}",
            instructions="You are a helpful test assistant. Keep responses brief.",
            model=Model(model_name)
        )

    agent = agents[model_name]

    try:
        response = await agent.send(message, timeout=120)
        return {
            "model": model_name,
            "message": message,
            "response": response[-1].content.text
        }
    except Exception as e:
        return {
            "model": model_name,
            "message": message,
            "error": str(e)
        }

Node.start(http_server=HttpServer(app=app))
