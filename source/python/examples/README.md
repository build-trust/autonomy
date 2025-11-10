To run the examples in this folder:

Make sure you have ollama running and serving models locally.

From inside `source/python` dirctory run:

```
AUTONOMY_USE_DIRECT_BEDROCK=1 \
AUTONOMY_WAIT_UNTIL_INTERRUPTED=0 \
AUTONOMY_USE_IN_MEMORY_DATABASE=1 \
CLUSTER="$(autonomy cluster show)" \
uv run --active examples/006.py
```
