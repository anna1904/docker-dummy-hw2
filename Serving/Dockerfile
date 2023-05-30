FROM huggingface/transformers-pytorch-gpu:4.22.1 as base

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install pip --upgrade
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install kserve==0.10.1 ray==2.0.0


RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app
COPY . .

CMD [ "bash" ]

# Streamlit UI docker image
FROM base AS app-streamlit
CMD streamlit run --server.address 0.0.0.0 --server.port 8080 serving/ui_app.py