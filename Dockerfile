FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements-streamlit.txt /tmp/requirements-streamlit.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements-streamlit.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.headless=true"]
