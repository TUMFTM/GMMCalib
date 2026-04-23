FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libgl1 \
        libgomp1 \
        git \
        git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN uv pip install --system --no-cache -r requirements.txt
EXPOSE 8000

CMD ["python", "src/gmmcalib.py"]