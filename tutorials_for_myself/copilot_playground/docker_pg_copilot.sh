# run docker container from ubuntu image
# docker run -it --rm --name pg_copilot -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:13.2-alpine
# docker run -it --rm ubuntu:18.04 bash
docker run -it --rm ruby:3.1.2 bash
docker run -it --rm ocaml/opam:latest bash