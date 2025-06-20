FROM python:3.11


WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/flask

EXPOSE 5003

CMD ["python", "prostate_app.py"]
