FROM v4tech/imagemagick
RUN mkdir -p /app/src /app/dest
COPY downsample.sh /app/
COPY group.sh /app/
COPY class_list.txt /app/

RUN apk update
RUN apk add --no-cache parallel bash
ENTRYPOINT ["/bin/bash"]
RUN ["/app/group.sh", "/app/src", "/app/dest"]
