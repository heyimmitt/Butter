FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# cpu only torch for now, swap to cuda index later when we set up gpu training
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements.txt && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless
    
COPY . .

CMD ["python", "test.py"]