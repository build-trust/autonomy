FROM ghcr.io/build-trust/autonomy-python-dev:latest AS dev
COPY requirements.txt ./
RUN pip install -r requirements.txt

FROM ghcr.io/build-trust/autonomy-python:latest
COPY --from=dev /app/venv venv
COPY main.py main.py
ENTRYPOINT ["python", "-m", "main"]
