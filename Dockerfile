FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        torch==2.14.1+cpu \
        torchvision==0.29.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt

COPY app.py ./
COPY benchmark_core ./benchmark_core

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py"]
