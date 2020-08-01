FROM python:3.6.11-slim
RUN apt-get update -y
# RUN apt-get install -y python3-pip python3-dev build-essential
COPY . /app
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.txt
# ENTRYPOINT ["python"]
CMD python /app/model.py && python /app/app.py
