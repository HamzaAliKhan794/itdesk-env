FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose required port
EXPOSE 7860

# Run FastAPI app natively
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
