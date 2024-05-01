FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ ./

RUN apt update && apt install unzip

CMD ["python", "kf_model_server.py"]
