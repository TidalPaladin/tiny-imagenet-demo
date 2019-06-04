FROM tensorflow/tensorflow:2.0.0a0-py3-jupyter as base
COPY tin /tin

FROM base as test
ARG cache=1
COPY test /test
RUN mkdir -p /cov
RUN pip install --no-cache-dir \
	pytest \
	pytest-cov \
	pytest-mock \
	pytest-timeout
ENTRYPOINT [ "pytest" ]
CMD [ "--cov=/tin", "--cov-report=xml:/cov/coverage.xml", "--cov-report=term", "/test"]
