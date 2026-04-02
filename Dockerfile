FROM nav2apo/zimage-api:latest

# Set working directory
WORKDIR /app

# Install only the new dependency needed for the edit pipelines.
RUN pip install --no-cache-dir "diffusers @ git+https://github.com/huggingface/diffusers.git"

# Overlay only the updated serverless entrypoints on top of the known-good image.
COPY handler.py zimage_wrapper.py /app/

# Expose port (not used in serverless, but kept for compatibility)
EXPOSE 8000

# Health check (not used in serverless, but kept for compatibility)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run RunPod serverless handler
CMD ["python", "handler.py"]
