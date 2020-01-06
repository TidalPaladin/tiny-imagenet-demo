ARG upstream=2.0.0rc0-gpu-py3
FROM tensorflow/tensorflow:${upstream} as base

# Expose Tensorboard ports
EXPOSE 6006/tcp 6006/udp

VOLUME [ "/data", "/artifacts" ]

COPY [ "docker/run.sh", "docker/entrypoint.sh", "/" ]
COPY requirements.txt /

RUN pip install -r /requirements.txt

COPY tin/ /app

WORKDIR /app
ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/run.sh" ]
