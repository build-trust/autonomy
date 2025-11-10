# Autonomy Documentation Project

This is a documentation project for **Autonomy** - a platform to build and deploy autonomous AI applications.

## 📁 What's in this directory

```
use/
├── CLAUDE.md                          # This file
├── _for-coding-agents.mdx             # Main reference guide for coding agents
├── _for-coding-agents/                # Topic-specific implementation guides
│   ├── create-a-new-autonomy-app.mdx  # Getting started, deployment, file structure
│   ├── select-models.mdx              # Available models and selection guide
│   ├── configure-apps.mdx             # Secrets, environment variables, dependencies
│   ├── create-custom-apis.mdx         # Custom FastAPI endpoints
│   ├── create-custom-ui.mdx           # Custom web interfaces (Next.js, React, etc.)
│   ├── memory.mdx                     # Memory, conversation, scope, user isolation
│   └── tools.mdx                      # Python tools and MCP tools
├── agents/                            # Agent-specific documentation
├── collaboration/                     # Collaboration and team features
├── for-llms/                          # LLM-specific documentation
├── integration/                       # Integration guides
├── start/                             # Getting started guides
└── demo-prompt.mdx                    # Demo and examples
```

## 🎯 Purpose

This documentation is designed for multiple audiences to use Autonomy effectively:

### For AI Coding Agents (Claude Code, Cursor, Cline, Zed, etc.)
- **`_for-coding-agents/`** - Implementation guides specifically designed for AI coding assistants
- Build Autonomy applications with AI agents
- Deploy apps to Autonomy Computer
- Find complete working examples and code templates

### For LLM Crawlers (ChatGPT, etc.)
- **`for-llms/`** - Documentation optimized for LLM crawlers and retrieval
- Reference documentation formatted for LLM consumption
- Best practices and patterns in crawler-friendly format

### For Human Developers
- **Other guides** (`agents/`, `collaboration/`, `integration/`, `start/`) - Human-readable documentation
- Learn to build and deploy autonomous AI applications
- Collaborate on Autonomy projects
- Integrate Autonomy with existing workflows

## 📝 About this project

This is a **documentation project** - the files here document how to build Autonomy applications, but this directory itself is not an Autonomy app.

When working on this project, you are:
- Writing and improving documentation
- Creating clear examples and code samples
- Organizing guides for different topics
- Ensuring consistency across documentation files

You are NOT building an Autonomy application in this directory.

## ✍️ Documentation structure

### For AI Coding Agents
- **`_for-coding-agents.mdx`** - Main entry point for AI coding agents with quick navigation
- **`_for-coding-agents/`** - Implementation guides for coding assistants (Claude Code, Cursor, Cline, etc.)

### For LLM Crawlers
- **`for-llms/`** - Documentation formatted for LLM crawlers (ChatGPT, etc.)

### For Human Developers
- **`agents/`** - Documentation focused on agent concepts and patterns
- **`collaboration/`** - Guides for team collaboration features
- **`integration/`** - Integration guides for connecting with external systems
- **`start/`** - Getting started guides for new users

### General Guidelines
- Each guide includes complete examples, commands, and best practices
- Guides cross-reference each other for related topics
- **All documentation files must end with an empty newline**

## 📖 Writing style guide

Write documentation appropriate for the target audience of each directory.

### Audience
- **`_for-coding-agents/`** - Target AI coding agents (Claude Code, Cursor, Cline, etc.) with technical competence but no prior Autonomy knowledge
- **`for-llms/`** - Optimize for LLM crawlers (ChatGPT, etc.) with structured, crawlable content
- **Other directories** - Target human developers with clear, comprehensive explanations
- Provide explicit and precise details - avoid ambiguity in all instructions

### Tone and Structure
- **Write in active voice** - Make the subject perform the action directly
  - ✅ "Create the file" not ❌ "The file should be created"
  - ✅ "Deploy the zone" not ❌ "The zone can be deployed"
  - ✅ "Run this command" not ❌ "This command should be run"
- **Use imperative mood** - Give direct commands
- **Front-load key information** - Place the most important details first
- **Follow consistent formatting** - Match patterns in existing docs
- **Use action-oriented section headers** - "Create...", "Deploy...", "Test..."
- **Start simple, then advance** - Show basic examples first, then more complex patterns

### File Structure (MDX frontmatter)
All `.mdx` files must include YAML frontmatter:

```yaml
---
title: "Coding Agents: How to [action]"
description: "Brief description of what this guide covers"
---
```

Optional fields:
- `mode: "wide"` - For wider layout
- Include keywords in italics after the main header

### Document Structure Pattern
Each guide should follow this structure:

1. **YAML frontmatter** - Title, description, optional mode
2. **Main header** - Repeats the title
3. **Keywords line** - In italics: `*Keywords: keyword1, keyword2, keyword3*`
4. **Key terms reference** - Blockquote pointing to main guide definitions:
   ```
   > **Key Terms**: See the [definitions section...
   ```
5. **Guide overview** - "This guide will:" followed by action-oriented bullets
6. **Main content** - Organized into clear sections with examples
7. **Related guides section** - Links to other relevant guides

### Code Examples
- **Include complete, working examples** - Never use pseudocode
- **Show full file paths** - Specify the exact directory structure
- **Include all necessary commands** - Never skip steps
- **Use realistic names** - Follow established conventions: "henry", "hello", "main-pod"
- **Add comments only to clarify non-obvious behavior**
- **Show file structure with tree diagrams** - Use simple indentation format

Example file structure format:
```
your-app/
├── autonomy.yaml
└── images/main/
    ├── Dockerfile
    └── main.py
```

### File Templates
- Provide exact, copy-paste-ready templates
- Include all required fields
- Show optional fields with comments
- Use consistent indentation (2 spaces for YAML, 4 for Python)
- Always wrap filenames and paths in backticks
- Always include an empty newline at the end of file templates

### Commands
- Include full commands with all flags
- Add `timeout` to curl commands for safety
- Show expected output when it helps understanding
- Provide both streaming and non-streaming examples
- Use `${CLUSTER}` and `${ZONE}` for placeholder variables
- Format multi-line commands with backslashes for readability

### Explanations After Code
After showing code examples, explain how they work:
- Use headers like "Key Concepts:", "Why This Works:", "How It Works:"
- Provide numbered or bulleted explanations
- Focus on the non-obvious aspects

### Cross-References
- Link to related guides at the end of each document
- Use consistent link text format: "Follow the [guide on topic](url)"
- Reference the main guide for definitions with blockquote format
- Always include "Related Guides" section at the end

### Keywords
- Include relevant keywords at the top of each guide
- Use keywords that coding agents might search for
- Cover synonyms and related terms
- Format: `*Keywords: keyword1, keyword2, keyword3*`

### Warnings and Notes
- Mark important callouts with `>` blockquotes
- Label critical information with **IMPORTANT** or **Note**
- Explain common pitfalls explicitly
- Use bold for emphasis on critical constraints

### Examples and Testing
- Always include a "Test..." section showing how to verify functionality
- Show both success cases and expected output
- Include curl commands with proper formatting
- Demonstrate progressive complexity (simple first, then advanced)

### Best Practices Sections
When including best practices:
- Use bullet points with specific, actionable advice
- Mark with ✅ for do's and ❌ for don'ts
- Keep them practical and implementation-focused

### What to Avoid
- ❌ Vague instructions ("configure as needed")
- ❌ Incomplete code snippets
- ❌ Assuming context from outside the docs
- ❌ Marketing language or fluff
- ❌ Outdated examples or deprecated patterns
- ❌ Skipping error handling in examples
- ❌ Defining `@app.get("/")` when you want automatic `index.html` serving