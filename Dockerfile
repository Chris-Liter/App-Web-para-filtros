# FROM python:alpine3.7
# COPY . /app 
# WORKDIR /app
# RUN pip install -r requeriments.txt
# EXPOSE 5000

# CMD ["python", "app.py"]

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Actualiza e instala dependencias ignorando fechas inválidas
RUN apt-get -o Acquire::Check-Valid-Until=false update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -o Acquire::Check-Valid-Until=false install -y \
    build-essential python3 python3-pip && \
    pip3 install flask pycuda numpy opencv-python

COPY . /app
WORKDIR /app
EXPOSE 5000

CMD ["python3", "app.py"]
