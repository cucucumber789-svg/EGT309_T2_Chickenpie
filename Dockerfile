FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY gas_monitoring.db.example .
CMD ["python", "src/model_rf.py"]
