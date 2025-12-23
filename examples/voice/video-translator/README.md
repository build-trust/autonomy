# Video Translator

Upload videos in any language → Get English transcriptions and translations.

This example demonstrates:
- **Whisper AI** for speech-to-text transcription (auto-detects language)
- **Claude** for translation to English
- **Box webhooks** for automatic processing of uploaded videos
- **Web UI** for direct upload demos

## Features

### 1. Box Integration (Production)
When a video is uploaded to your Box folder:
1. Box sends a webhook notification to your app
2. App downloads the video and extracts audio
3. Whisper transcribes the audio (auto-detects language)
4. Claude translates the transcription to English
5. Results are uploaded back to Box as a markdown file

### 2. Demo UI (Direct Upload)
Visit the app URL to upload videos directly through a web interface.

## Setup

### Prerequisites
- [Autonomy CLI](https://autonomy.computer/docs/get-started) installed
- Box Developer account (for Box integration)

### 1. Clone and configure

```bash
cd examples/voice/video-translator
cp secrets.yaml.example secrets.yaml
```

### 2. Configure Box (optional, for webhook integration)

1. Go to [Box Developer Console](https://app.box.com/developers/console)
2. Create a new **Custom App** with **Server Authentication (Client Credentials Grant)**
3. Under Configuration, note your:
   - Client ID
   - Client Secret  
   - Enterprise ID
4. Edit `secrets.yaml` with your credentials:

```yaml
BOX_CLIENT_ID: "your_client_id"
BOX_CLIENT_SECRET: "your_client_secret"
BOX_ENTERPRISE_ID: "your_enterprise_id"
BOX_FOLDER_ID: "your_folder_id"
WEBHOOK_BASE_URL: "https://your-cluster-vidtrans.cluster.autonomy.computer"
```

### 3. Deploy

```bash
autonomy zone deploy
```

### 4. Get your app URL

```bash
autonomy cluster show
```

Your app will be available at:
```
https://${CLUSTER}-vidtrans.cluster.autonomy.computer
```

### 5. Configure Box Webhook (if using Box integration)

After deployment, the app will automatically try to register a webhook with Box.
Alternatively, you can manually set up a webhook in Box:

1. In Box Developer Console, go to your app's Webhooks tab
2. Create a webhook:
   - **URL**: `https://${CLUSTER}-vidtrans.cluster.autonomy.computer/webhook/box`
   - **Target**: Your videos folder
   - **Triggers**: `FILE.UPLOADED`

## Usage

### Demo UI

1. Open `https://${CLUSTER}-vidtrans.cluster.autonomy.computer` in your browser
2. Drag and drop a video file (or click to browse)
3. Wait for processing (extracting audio → transcribing → translating)
4. View results and download the translation

### Box Integration

1. Upload a video to your configured Box folder
2. The app automatically processes it
3. A `{filename}_translation.md` file appears in the same folder

### API Endpoints

```bash
# Direct upload
curl -X POST -F "file=@video.mp4" \
  https://${CLUSTER}-vidtrans.cluster.autonomy.computer/upload

# Health check
curl https://${CLUSTER}-vidtrans.cluster.autonomy.computer/health

# Webhook status
curl https://${CLUSTER}-vidtrans.cluster.autonomy.computer/webhook/status
```

## Supported Video Formats

- MP4, MOV, AVI, MKV, WebM, M4V, FLV, WMV

## How It Works

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│   Video     │───▶│    FFmpeg    │───▶│   Whisper   │───▶│   Claude   │
│   Upload    │    │ Extract Audio│    │ Transcribe  │    │ Translate  │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
                                              │                  │
                                              ▼                  ▼
                                        Original text      English text
```

1. **Audio Extraction**: FFmpeg extracts the audio track from the video
2. **Transcription**: Whisper AI transcribes the audio, auto-detecting the language
3. **Translation**: Claude translates the transcription to English
4. **Results**: Both original transcription and English translation are returned

## Project Structure

```
video-translator/
├── autonomy.yaml           # Deployment configuration
├── secrets.yaml.example    # Template for credentials
├── secrets.yaml            # Your credentials (gitignored)
└── images/
    └── main/
        ├── Dockerfile      # Container with ffmpeg
        ├── main.py         # FastAPI app with endpoints
        ├── box.py          # Box SDK client
        ├── index.html      # Demo upload UI
        └── requirements.txt
```

## Troubleshooting

### "No audio track found"
The video file doesn't contain an audio track. Try a different video.

### "No speech detected"
The audio track exists but Whisper couldn't detect any speech. This might happen with:
- Silent videos
- Videos with only music/sound effects
- Very poor audio quality

### Box webhook not triggering
1. Check that your webhook URL is correct and publicly accessible
2. Verify the webhook is registered: `curl .../webhook/status`
3. Check Box webhook logs in the Developer Console

### Processing timeout
Large videos may take several minutes. The default timeout is 3 minutes for translation.
For very long videos, consider splitting them first.