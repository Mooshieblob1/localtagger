# LAN Image Interrogator

A unified FastAPI microservice for **EVA-CLIP** and **WD14 (PixAI)** image interrogation, designed for Kubernetes deployment with GPU support.

## Features
-   **Multi-Model Support**:
    -   **VIT** (ViT-L-14/openai): Standard CLIP.
    -   **EVA** (ViT-g-14/laion2b_s12b_b42k): High-accuracy CLIP.
    -   **PixAI** (WD14): Anime/Illustration tagging.
-   **On-Demand Loading**: Optimizes VRAM by dynamically loading/unloading large models.
-   **RESTful API**: Simple POST (file) and GET (URL) endpoints.
-   **Kubernetes Ready**: Includes health checks and manifests for GPU deployment.

## Kubernetes Deployment (Recommended)

### 1. Build & Push Image
Build the Docker image and push it to your container registry (GHCR, Docker Hub, internal registry).

```bash
# Example using GHCR
docker build -t ghcr.io/mooshieblob1/localtagger:latest .
docker push ghcr.io/mooshieblob1/localtagger:latest
```

### 2. Deploy
The `k8s/` directory contains standard manifests.

1.  **Update Image**: Edit `k8s/deployment.yaml` to point to your pushed image if different from the default.
2.  **Apply Manifests**:
    ```bash
    kubectl apply -f k8s/
    ```

### Deployment Notes
-   **GPU Requirements**: The deployment requests `nvidia.com/gpu: 1`. Ensure your nodes have NVIDIA drivers and the **NVIDIA Container Toolkit** installed.
-   **Model Caching**: Models are downloaded to `/root/.cache/huggingface` on first use (approx 5GB+). To improve startup times and avoid repeated downloads, mount a **Persistent Volume (PVC)** to this path.
-   **Health Check**: The service exposes `GET /health` for Liveness and Readiness probes.

---

## API Usage

The service exposes a unified endpoint for image interrogation using the **WD14 (EVA02-Large)** model.

### Main Endpoint: `/interrogate`

Supports both single image and batch processing.

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | File(s) | Required | Image file(s) to process. Send multiple for batching. |
| `output_format` | string | `"json"` | `"zip"` for dataset download, `"json"` for API response. |
| `threshold` | float | `0.35` | Confidence threshold (0.0 - 1.0). |
| `trigger_word` | string | `""` | Optional word to prepend to tags (e.g., `sks_person`). |
| `random_order` | boolean | `false` | Shuffle tags (recommended for LoRA training). |
| `use_spaces` | boolean | `false` | Use spaces instead of underscores in tags. |
| `use_escape` | boolean | `true` | Escape special characters (parentheses). |

---

## Frontend Integration Guide

### 1. Response Structure (JSON)

When `output_format="json"` (default), the API returns an **Array of Objects**.

```json
[
  {
    "tags": {
      "general": 0.99,
      "cat": 0.97,
      "animal": 0.82
    },
    "tag_string": "general, cat, animal"
  }
]
```

**Key Fields:**
*   **`tags`**: Object with tag names as keys and confidence scores (0.0-1.0) as values.
*   **`tag_string`**: Comma-separated string of tags, ready for display or text files.

### 2. Batch Processing (Multiple Images)

To process multiple images, append them to the same form field name `file`.

**JavaScript Example:**
```javascript
const formData = new FormData();
files.forEach((file) => {
  formData.append("file", file); // Must be "file" for all images
});

// Add query parameters
const params = new URLSearchParams({
  threshold: "0.35",
  random_order: "true"
});

await fetch(`http://localhost:8000/interrogate?${params}`, {
  method: "POST",
  body: formData
});
```

### 3. Downloading Datasets (ZIP)

To generate a dataset for LoRA training, set `output_format="zip"`. The response will be a binary blob.

**JavaScript Example:**
```javascript
const params = new URLSearchParams({
  output_format: "zip",
  trigger_word: "my_trigger",
  random_order: "true"
});

const response = await fetch(`http://localhost:8000/interrogate?${params}`, {
  method: "POST",
  body: formData
});

if (response.ok) {
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "dataset.zip";
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
}
```

---

## Local Development

You can run the service locally using Docker.

```bash
# Build and Run
./build_and_run.sh

# Or manually:
docker build -t lan-interrogator .
docker run --gpus all -d -p 8000:8000 --name interrogator lan-interrogator
```
