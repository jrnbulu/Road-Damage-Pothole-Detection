# CPU inference image for the API (default) or the Streamlit app.
#   docker build -t pothole-detector .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models" pothole-detector              # API
#   docker run -p 8501:8501 -v "$PWD/models:/app/models" pothole-detector \
#       streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0   # app
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/best.pt YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pothole/ pothole/
COPY api/ api/
COPY app/ app/
COPY samples/ samples/
COPY models/ models/

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
