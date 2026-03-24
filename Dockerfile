FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files first (cache layer)
COPY pyproject.toml .
RUN uv pip install --system -e .

# Copy source
COPY src/ src/
COPY sample-data/ sample-data/

# Create output directory
RUN mkdir -p reports

ENTRYPOINT ["python", "-m", "src.main"]
