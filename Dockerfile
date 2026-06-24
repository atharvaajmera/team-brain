FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Command to run bot or sync based on arg
ENTRYPOINT ["python"]
CMD ["slack_bot.py"]
