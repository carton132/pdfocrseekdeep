# DeepSeek PDF OCR

A REST API service for converting PDFs and images to accessible markdown using [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR).

Designed to run on RunPod with model weights stored on a network volume for fast pod spin-up.

## Features

- 📄 **PDF to Markdown** - Full document conversion with layout preservation
- 🖼️ **Image OCR** - Process individual images
- 📊 **Table Detection** - Structured table extraction to HTML/markdown
- 📈 **Figure Parsing** - Describe charts and diagrams
- 🚀 **Fast Setup** - Model on network volume, app cloned fresh each pod
- 🔌 **REST API** - Easy integration with any client

## Architecture

```
Network Volume (/workspace)          Container (ephemeral)
├── models/deepseek-ocr/  (7GB)     ├── app/deepseek-pdf-ocr/
├── cache/pip/                       │   ├── server.py
├── config.env                       │   └── ...
├── setup.sh                         └── workdir/
└── logs/                                ├── input/
                                         └── output/
```

## Quick Start (RunPod)

### First Time Setup

1. **Create a RunPod pod** with:
   - GPU: A40, L40S, or A100 (40GB+ VRAM recommended)
   - Image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - Network Volume: Attach at `/workspace`

2. **SSH into the pod** and run:
   ```bash
   # Download setup script (first time only)
   curl -o /workspace/setup.sh https://raw.githubusercontent.com/YOUR_USERNAME/deepseek-pdf-ocr/main/setup.sh
   
   # Edit to set your repo URL
   nano /workspace/setup.sh
   
   # Run setup (downloads model ~7GB on first run)
   bash /workspace/setup.sh
   ```

3. **Start the server:**
   ```bash
   bash /workspace/setup.sh --start-server
   ```

### Subsequent Pods

Once model is downloaded, new pods are ready in ~2 minutes:

```bash
bash /workspace/setup.sh --start-server
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### OCR an Image

```bash
curl -X POST http://localhost:8000/ocr/image \
  -F "file=@document.png" \
  -F "mode=document" \
  -F "resolution=base"
```

### OCR a PDF

```bash
curl -X POST http://localhost:8000/ocr/pdf \
  -F "file=@document.pdf" \
  -F "mode=document" \
  -F "resolution=base"
```

### API Documentation

Interactive docs available at: `http://localhost:8000/docs`

## Configuration

Edit `/workspace/config.env` to customize:

```bash
# Resolution: tiny, small, base, large, gundam
DEFAULT_RESOLUTION=base

# PDF rendering quality
PDF_DPI=300

# Output format
DEFAULT_OUTPUT_FORMAT=markdown
```

## OCR Modes

| Mode | Prompt | Use Case |
|------|--------|----------|
| `document` | Convert to markdown with layout | Standard documents |
| `free` | Raw OCR without structure | Simple text extraction |
| `figure` | Parse the figure | Charts, diagrams |
| `table` | Convert with table focus | Table-heavy pages |
| `describe` | Describe in detail | Image descriptions |

## Resolution Settings

| Setting | Size | Tokens | Use Case |
|---------|------|--------|----------|
| `tiny` | 512×512 | 64 | Fast preview |
| `small` | 640×640 | 100 | Quick processing |
| `base` | 1024×1024 | 256 | **Recommended default** |
| `large` | 1280×1280 | 400 | High detail |
| `gundam` | Dynamic | Variable | Complex documents |

## Development

### Running Tests

```bash
cd /root/app/deepseek-pdf-ocr
python test_setup.py           # Full test with inference
python test_setup.py --quick   # Skip inference test
```

### Local Development (Mac)

The server runs on RunPod; develop the client locally:

```python
import requests

# Upload and process PDF
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://YOUR_POD_IP:8000/ocr/pdf",
        files={"file": f},
        data={"mode": "document"}
    )

result = response.json()
print(result["content"])  # Markdown output
```

## Troubleshooting

### Model not loading
- Check GPU memory: `nvidia-smi`
- Verify model files: `ls /workspace/models/deepseek-ocr/`
- Re-download: `bash /workspace/setup.sh --fresh`

### Out of memory
- Use smaller resolution: `resolution=small`
- Process fewer pages at once: `start_page=1&end_page=10`

### Flash attention errors
- Falls back to standard attention automatically
- Or set `USE_MEMORY_EFFICIENT_ATTENTION=true` in config.env

## License

MIT License - See LICENSE file

## Acknowledgments

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) by DeepSeek AI
- [RunPod](https://runpod.io) for GPU infrastructure
