FROM python:3.8-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libgl1-mesa-glx \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000

CMD ["python", "src/gmmcalib.py"]