FROM python:3.12-slim as latest

# Set working directory
WORKDIR /app
COPY requirements.txt /app
# Install dependencies (CPU-only PyTorch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt  --root-user-action=ignore
COPY . .

FROM latest as final
COPY . . 
CMD ["python", "app.py"]

