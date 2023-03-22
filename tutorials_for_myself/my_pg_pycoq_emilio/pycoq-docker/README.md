### docker image for running pycoq + conda

This Dockerfile installs the correct version of all required packages for running pycoq with conda. The default conda env is 'pycoq'.

To build the image (expect 20mins+ on M1 Max):
```bash
docker build --platform linux/amd64 -t coq-image .
```

To run the image:
```bash
docker run --name=coq -it coq-image
# or to mount the current directory
# $ docker run --name=coq -it -v "$PWD:$PWD" -w "$PWD" coq-image
```

To restart the image:
```bash
docker start -ai coq
```

To remove the image:
```bash
docker rm coq
```