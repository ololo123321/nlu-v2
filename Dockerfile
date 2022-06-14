FROM nvidia/cuda:10.0-runtime-ubuntu18.04
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3.7-venv python3-pip curl vim wget
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --upgrade cython
RUN python -m pip install -r requirements.txt
RUN wget https://github.com/conll/reference-coreference-scorers/archive/v8.01.tar.gz \
    && tar -xvzf v8.01.tar.gz