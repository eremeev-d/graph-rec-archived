FROM python:3.10.13

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir /app
COPY app app

RUN mkdir /data
COPY data data

WORKDIR /app

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]