COPY lid.176.bin /app/

FROM python:3.9-slim as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \  # Required for PyTorch
    && rm -rf /var/lib/apt/lists/*
    
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . /app
WORKDIR /app

CMD ["python", "bot.py"]
