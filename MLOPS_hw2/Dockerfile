FROM python:3.11

WORKDIR /app

COPY ./requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 4000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=4000"]