FROM continuumio/miniconda:4.5.4

RUN pip install mlflow>=1.5.0 \
    && pip install numpy==1.16.1 \
    && pip install sklearn \
    && pip install keras \
    && pip install tensorflow
