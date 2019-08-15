ARG upstream=2.0.0b1-gpu-py3
FROM tensorflow/tensorflow:${upstream} as base

VOLUME [ "/data", "/artifacts" ]

# Expose Tensorboard ports
EXPOSE 6006/tcp 6006/udp

COPY [ "run.sh", "entrypoint.sh", "/" ]
COPY tin/ /app

WORKDIR /app
ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/run.sh" ]
