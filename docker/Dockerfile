FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
WORKDIR /app
COPY ./requirements.txt .
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install uvicorn[standard]