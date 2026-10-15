FROM python:3.14-slim
WORKDIR /app

# Copy requirements and install normally (cached automatically by Docker layers)
COPY requirements.txt .
RUN pip install --root-user-action=ignore -r requirements.txt

# Copy application code last
COPY . .

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "v22_main:app", "--host", "0.0.0.0", "--port", "8080"]
