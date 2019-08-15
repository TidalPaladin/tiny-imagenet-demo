FROM v4tech/imagemagick

RUN apk update
RUN apk add --no-cache parallel bash

VOLUME ["/data/src", "/data/dest"]

COPY src /app
WORKDIR /app

ENTRYPOINT ["/bin/bash"]
CMD ["/app/group.sh", "/data/src", "/data/dest"]
