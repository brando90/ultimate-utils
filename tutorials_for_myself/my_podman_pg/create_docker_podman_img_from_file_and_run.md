#

install podman on mmac
```
brew install podman
podman machine init
podman machine start
```
build docker img from dockerfile
```
podman build -t iit .
```
run container
```
podman run -it iit /bin/bash
```

## linux 

install: https://podman.io/getting-started/installation