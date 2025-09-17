### English to Hindi Translation API

This example demonstrates how to create a translation API using Autonomy that translates English text to Hindi (written in Latin alphabet).

### Install

First, make sure you have the `autonomy` command installed. Follow the installation instructions in the main Autonomy documentation.

### Get code from this example

Navigate to this directory:

```sh
cd autonomy/examples/004
```

### Populate secrets

The translation API is protected with an API key that must be provided in a `secrets.yaml` file.
Use the included `secrets.example.yaml` file as a template.

Create `secrets.yaml` with a strong random 32-byte key:

```sh
sed "s|your_api_key_here|$(openssl rand -hex 32)|" secrets.example.yaml > secrets.yaml
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
- HTTP API server at `http://localhost:32100`
- Logs at `http://localhost:32101`

### Test the API

Once the service is running, set up your environment variables:

```sh
export URL='http://localhost:32100'
export API_KEY='<your_api_key_from_secrets.yaml>'
```

Test the translation API using curl:

```sh
curl -s -X POST "$URL/analyses" \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"items":["hello", "goodbye", "thank you"]}'
```

The API accepts a list of English text items and returns translations in Hindi written using the Latin alphabet. Each item is processed in parallel for better performance.

### API Response Format

The response will be a JSON object containing an `analyses` array with translations:

```json
{
  "analyses": [
    {"item": "hello", "analysis": "namaste"},
    {"item": "goodbye", "analysis": "alvida"},
    {"item": "thank you", "analysis": "dhanyawad"}
  ]
}
```

If any translation fails, the response will include an error field instead of analysis:

```json
{
  "item": "problematic text",
  "error": "error description"
}
```
