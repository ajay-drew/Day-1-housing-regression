FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY housing-regression-model.py .
RUN mkdir -p /app/data
ENV PYTHONUNBUFFERED=1
CMD ["python", "housing-regression-model.py"]