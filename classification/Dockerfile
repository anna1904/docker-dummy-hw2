FROM pytorch/pytorch:latest

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app
COPY . .

CMD [ "bash" ]