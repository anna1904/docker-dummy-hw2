FROM jupyter/minimal-notebook

RUN pip install pandas

COPY main.ipynb .

EXPOSE 1337

CMD ["jupyter", "notebook", "--port=1337"]