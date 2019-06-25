FROM tensorflow/tensorflow:2.0.0b0-py3-jupyter as base
COPY tin /app/tin
COPY train.py /app/
ENTRYPOINT [ "python" ]
CMD [ "/app/train.py" ]

FROM base as test
ARG cache=1
COPY test /app/test
RUN mkdir -p /app/cov
RUN pip install --no-cache-dir \
	pytest \
	pytest-cov \
	pytest-mock \
	pytest-timeout
ENTRYPOINT [ "pytest" ]
CMD [ "--cov=/app/tin", "--cov-report=xml:/app/cov/coverage.xml", "--cov-report=term", "/app/test"]
