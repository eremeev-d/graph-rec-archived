FROM python:3.10.13

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir /data
COPY data data

RUN mkdir /app
COPY app app

EXPOSE 8501

ENV PYTHONPATH .

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501"]