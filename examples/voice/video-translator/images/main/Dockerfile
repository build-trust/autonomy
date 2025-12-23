FROM ghcr.io/build-trust/autonomy-python-dev

# Install ffmpeg for audio extraction (as root)
USER root
RUN apk add --no-cache ffmpeg
USER nonroot

# Install Python dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application code
COPY . .

ENTRYPOINT ["python", "main.py"]
