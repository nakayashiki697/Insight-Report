# Insight Report - Hugging Face Spaces Dockerfile
# https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

# WeasyPrint dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    gir1.2-gdkpixbuf-2.0 \
    libffi-dev \
    shared-mime-info \
    fonts-noto-cjk \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Create necessary directories with correct permissions
RUN mkdir -p uploads outputs temp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV HF_HOME=/app/.cache

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Run with Gunicorn (single worker for session/progress consistency)
CMD ["gunicorn", "app.main:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300"]

