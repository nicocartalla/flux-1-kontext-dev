#FROM diffusers/diffusers-pytorch-cuda
FROM docker.io/rayproject/ray:2.37.0-gpu

ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cu121 \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    HF_HOME=/tmp/hf-cache \
    PYTHONUNBUFFERED=1


COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY server.py /app/server.py