FROM v4tech/imagemagick
RUN apk update
RUN apk add --no-cache parallel bash

RUN mkdir -p /data/src /data/dest
COPY src /app
COPY run.sh /
WORKDIR /app

ENTRYPOINT ["/bin/bash"]
CMD ["/run.sh", "/data/src", "/data/dest"]
