ARG upstream=2.0.0b1-gpu-py3
FROM tensorflow/tensorflow:${upstream} as base

RUN pip install Pillow==6.1.0 scipy==1.3.1

VOLUME [ "/data", "/artifacts" ]

# Expose Tensorboard ports
EXPOSE 6006/tcp 6006/udp

COPY [ "docker/run.sh", "docker/entrypoint.sh", "/" ]
COPY tin/ /app

WORKDIR /app
ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/run.sh" ]
