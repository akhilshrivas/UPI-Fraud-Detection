FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY upi_fraud ./upi_fraud
RUN python -m upi_fraud.generate && python -m upi_fraud.train
EXPOSE 8000
CMD ["uvicorn", "upi_fraud.api:app", "--host", "0.0.0.0", "--port", "8000"]
