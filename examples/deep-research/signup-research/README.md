# Signup Research - Autonomy App

An AI-powered application that researches new users signing up for your product. When a user signs up, you provide their name, email, and organization, and an AI agent researches them using the Linkup web search API to provide relevant context about their company and how your product could help them.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Autonomy Computer                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                              Zone: research                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                          main-pod                                │  │  │
│  │  │                                                                  │  │  │
│  │  │   ┌──────────────────────────────────────────────────────────┐  │  │  │
│  │  │   │                    main container                         │  │  │  │
│  │  │   │                                                           │  │  │  │
│  │  │   │  ┌─────────────────────────────────────────────────────┐  │  │  │  │
│  │  │   │  │                  Autonomy Node                       │  │  │  │  │
│  │  │   │  │                                                      │  │  │  │  │
│  │  │   │  │  ┌────────────────┐    ┌─────────────────────────┐  │  │  │  │  │
│  │  │   │  │  │   FastAPI      │    │   Research Agent        │  │  │  │  │  │
│  │  │   │  │  │   HTTP Server  │───▶│   (Claude Sonnet 4)     │  │  │  │  │  │
│  │  │   │  │  │                │    │                         │  │  │  │  │  │
│  │  │   │  │  │  /             │    │  ┌───────────────────┐  │  │  │  │  │  │
│  │  │   │  │  │  (Next.js UI)  │    │  │  Python Tools     │  │  │  │  │  │  │
│  │  │   │  │  │                │    │  │                   │  │  │  │  │  │  │
│  │  │   │  │  │  /api/research │    │  │  • linkup_search  │  │  │  │  │  │  │
│  │  │   │  │  │                │    │  │  • linkup_fetch   │  │  │  │  │  │  │
│  │  │   │  │  └────────────────┘    │  └─────────┬─────────┘  │  │  │  │  │  │
│  │  │   │  │                        │            │            │  │  │  │  │  │
│  │  │   │  └────────────────────────┴────────────┼────────────┘  │  │  │  │  │
│  │  │   │                                        │               │  │  │  │  │
│  │  │   └────────────────────────────────────────┼───────────────┘  │  │  │  │
│  │  │                                            │                  │  │  │  │
│  │  └────────────────────────────────────────────┼──────────────────┘  │  │  │
│  │                                               │                     │  │  │
│  └───────────────────────────────────────────────┼─────────────────────┘  │  │
│                                                  │                        │  │
└──────────────────────────────────────────────────┼────────────────────────┘  │
                                                   │                           
                                                   ▼                           
                                     ┌─────────────────────────┐               
                                     │     Linkup API          │               
                                     │  (Web Search Service)   │               
                                     │                         │               
                                     │  • Search web content   │               
                                     │  • Fetch web pages      │               
                                     │  • Extract information  │               
                                     └─────────────────────────┘               
```

## Data Flow

```
┌──────────┐     ┌──────────────┐     ┌─────────────────┐     ┌────────────┐
│  User    │────▶│  Next.js UI  │────▶│  FastAPI        │────▶│  Research  │
│  Input   │     │  (Browser)   │     │  /api/research  │     │  Agent     │
└──────────┘     └──────────────┘     └─────────────────┘     └─────┬──────┘
                                                                    │
     ┌──────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Agent Research Process                           │
│                                                                          │
│  1. Parse user info (name, email, organization)                         │
│                           │                                              │
│                           ▼                                              │
│  2. Call linkup_search() - Search for company information               │
│     • Value proposition                                                  │
│     • Products/services                                                  │
│     • Recent news                                                        │
│                           │                                              │
│                           ▼                                              │
│  3. Call linkup_fetch() - Get detailed content from relevant pages      │
│                           │                                              │
│                           ▼                                              │
│  4. Synthesize findings and generate personalized insights              │
│     • Company summary                                                    │
│     • How Linkup could benefit them                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **User Research**: Automatically research new signups using web search
- **Company Intelligence**: Gather info about company products, value prop, and news
- **Personalized Insights**: Generate tailored explanations of how Linkup can help
- **Streaming Responses**: Real-time updates as research progresses
- **Modern UI**: Clean Next.js interface built with shadcn/ui

## Tech Stack

- **Backend**: Autonomy Framework (Python) with FastAPI
- **Agent**: Claude Sonnet 4 via Autonomy
- **Search**: Linkup Python SDK for web search
- **Frontend**: Next.js 14 with shadcn/ui components
- **Deployment**: Autonomy Computer

## Project Structure

```
signup-research/
├── autonomy.yaml           # Zone configuration
├── secrets.yaml            # API keys (gitignored)
├── .gitignore
├── README.md
├── images/
│   └── main/
│       ├── Dockerfile      # Multi-stage build
│       ├── requirements.txt
│       ├── main.py         # Agent + FastAPI server
│       └── public/         # Compiled Next.js UI
└── ui/                     # Next.js source
    ├── package.json
    ├── next.config.js
    ├── tailwind.config.js
    ├── components.json
    └── src/
        ├── app/
        │   ├── layout.tsx
        │   ├── page.tsx
        │   └── globals.css
        ├── components/
        │   └── ui/
        └── lib/
            └── utils.ts
```

## Setup

### Prerequisites

- Docker installed and running
- Autonomy CLI installed (`curl -sSfL autonomy.computer/install | bash`)
- Linkup API key ([Get one here](https://app.linkup.so))
- Autonomy account ([Sign up here](https://my.autonomy.computer))

### Configuration

1. Create `secrets.yaml` with your Linkup API key:

```yaml
LINKUP_API_KEY: "your_linkup_api_key_here"
```

2. Enroll with your Autonomy cluster:

```bash
autonomy cluster enroll --no-input
```

### Build UI

```bash
cd ui
npm install
npm run build-autonomy
```

### Deploy

```bash
autonomy zone deploy
```

### Access

After deployment (wait ~2-3 minutes for cold start):

- **UI**: `https://${CLUSTER}-research.cluster.autonomy.computer/`
- **API**: `https://${CLUSTER}-research.cluster.autonomy.computer/api/research`

## Usage

1. Open the web UI
2. Enter the new user's information:
   - **Name**: Full name of the person
   - **Email**: Their email address
   - **Organization**: Company name
3. Click "Research"
4. View the streaming results including:
   - Company overview and value proposition
   - Products and services
   - Recent news and updates
   - How Linkup could benefit their organization

## API

### POST /api/research

Request:
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "organization": "Example Corp"
}
```

Response (streaming):
```json
{"scope": "...", "conversation": "...", "messages": [...], "type": "conversation_snippet"}
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LINKUP_API_KEY` | Linkup API key for web search | Yes |

## License

MIT