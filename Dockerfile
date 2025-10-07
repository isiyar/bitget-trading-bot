FROM python:3.13
WORKDIR /app
COPY .env .
COPY tg_to_bitget_autolong.py .
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "tg_to_bitget_autolong.py"]