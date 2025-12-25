FROM zimage-base

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (not used in serverless, but kept for compatibility)
EXPOSE 8000

# Health check (not used in serverless, but kept for compatibility)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run RunPod serverless handler
CMD ["python", "handler.py"]
