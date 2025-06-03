FROM python:3.10
RUN mkdir /app
COPY . /app

WORKDIR  /app

RUN chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda 

ENV PATH="/opt/miniconda/bin:${PATH}"
RUN conda init bash
RUN /bin/bash -c "source /root/.bashrc"
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN export MODEL_WEIGHTS_PATH=/app/models/finetuned-model/m3e-base

#EXPOSE 5000
#HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
#   CMD curl --fail http://localhost:5000/health || exit 1
RUN chmod +x /app/start_server.sh
CMD ["/app/start_server.sh"]
