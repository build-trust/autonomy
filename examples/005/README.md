### Legal Assistant - Henry

This example demonstrates how to create a legal assistant chatbot using Autonomy with a Knowledge base loaded with US Federal Code documents. The assistant provides a web interface for interactive conversations about weather modification legal requirements.

### Install

First, make sure you have the `autonomy` command installed. Follow the installation instructions in the main Autonomy documentation.

### Get code from this example

Navigate to this directory:

```sh
cd autonomy/examples/005
```

### Run the example

While actively developing, you can run a temporary deployment that automatically reloads on changes:

> [!IMPORTANT]
> Ensure you have [Docker](https://www.docker.com/get-started/) installed
> and running on your workstation before running the following command.

```sh
autonomy --rm
```

This will start the service and output URLs for:
- Web interface at `http://localhost:32100`
- Logs at `http://localhost:32101`

### Use the Legal Assistant

Once the service is running, open your browser and navigate to `http://localhost:32100` to access the chat interface.

Henry, the legal assistant, has been loaded with knowledge about US Federal Code Title 15, Chapter 9A covering weather modification activities and reporting requirements. You can ask questions such as:

- "What are the reporting requirements for weather modification activities?"
- "Who needs to submit reports under Section 330a?"
- "What penalties exist for non-compliance?"
- "What information must be included in weather modification reports?"

### Features

- **Interactive Chat Interface**: Clean, terminal-style web interface for conversations
- **Streaming Responses**: Real-time character-by-character response streaming
- **Knowledge-Enhanced**: Pre-loaded with relevant US Federal Code documents
- **Legal Expertise**: Specialized instructions for legal assistance

### Knowledge Base

The assistant is loaded with the following US Federal Code sections:

- **Section 330**: Weather modification activities; reporting requirement
- **Section 330a**: Report requirement; form; information
- **Section 330b**: Duties of Secretary
- **Section 330c**: Authority of Secretary
- **Section 330d**: Violation; penalty
- **Section 330e**: Authorization of appropriations

These documents are automatically fetched and processed when the service starts, enabling Henry to provide accurate information about weather modification legal requirements.

### Technical Details

- **Model**: Uses `nova-micro-v1` for efficient and responsive conversations
- **Knowledge Integration**: Documents are loaded from GitHub raw URLs and processed as markdown
- **Web Interface**: Clean HTML interface served automatically by Autonomy
- **Streaming**: Supports real-time response streaming for better user experience
