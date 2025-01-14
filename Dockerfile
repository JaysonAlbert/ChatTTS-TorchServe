FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    openjdk-17-jdk \
    make --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "numpy<2.0.0" torchserve torch-model-archiver  ChatTTS nvgpu soundfile nemo_text_processing WeTextProcessing --no-cache-dir


COPY ./model_store /app/model_store
COPY ./config.properties /app/config.properties

CMD ["torchserve", "--start", "--model-store", "/app/model_store", "--models", "chattts=chattts.mar", "--ts-config", "/app/config.properties", "--foreground"]
