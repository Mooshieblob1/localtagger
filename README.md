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

The service exposes endpoints for three models. You can upload an image file (`POST`) or provide an image URL (`GET`).

### 1. VIT (ViT-L-14/openai)
Standard CLIP model. Good balance of speed and accuracy.
-   **POST (Upload):**
    ```bash
    curl -X POST -F "file=@image.jpg" "http://<SERVICE-IP>/interrogate/vit?mode=fast"
    ```
-   **GET (URL):**
    ```bash
    curl "http://<SERVICE-IP>/interrogate/vit?url=https://example.com/image.jpg&mode=fast"
    ```

### 2. EVA (ViT-g-14/laion2b_s12b_b42k)
High-performance CLIP model. **Note:** First request will take time to swap models (~30s).
-   **POST (Upload):**
    ```bash
    curl -X POST -F "file=@image.jpg" "http://<SERVICE-IP>/interrogate/eva?mode=best"
    ```
-   **GET (URL):**
    ```bash
    curl "http://<SERVICE-IP>/interrogate/eva?url=https://example.com/image.jpg&mode=best"
    ```

### 3. PixAI (WD14 - Anime Tags)
Specialized for anime/illustration tagging.
-   **POST (Upload):**
    ```bash
    curl -X POST -F "file=@image.jpg" "http://<SERVICE-IP>/interrogate/pixai?threshold=0.35"
    ```
-   **GET (URL):**
    ```bash
    curl "http://<SERVICE-IP>/interrogate/pixai?url=https://example.com/image.jpg&threshold=0.35"
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
