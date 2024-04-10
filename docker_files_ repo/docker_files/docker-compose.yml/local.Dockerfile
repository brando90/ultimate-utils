FROM alpine:3.10 AS preparer

RUN apk add bash curl

RUN curl -sSL https://git.io/get-mo -o mo
RUN chmod +x mo

ARG HOST
ENV HOST $HOST
ARG IDP
ENV IDP $IDP

COPY etc-httpd/conf.d/sp.conf .
COPY etc-shibboleth/idp-metadata.xml .
COPY etc-shibboleth/local.shibboleth2.xml .

RUN mkdir out
RUN ./mo sp.conf > out/sp.conf
RUN ./mo idp-metadata.xml > out/idp-metadata.xml
RUN ./mo local.shibboleth2.xml > out/shibboleth2.xml

ARG BASE_IMAGE=local/shibboleth-sp:3.0.4
FROM ${BASE_IMAGE} AS final-stage

RUN yum -y update \
    && yum -y install php mod_ssl

COPY etc-shibboleth/ /etc/shibboleth/
COPY etc-httpd/ /etc/httpd/
# Adjustments for local or specific configurations
RUN rm /etc/httpd/conf.d/ssl.conf
COPY var-www-html/ /var/www/html/

# Copy files from the "preparer" stage
COPY --from=preparer out/sp.conf /etc/httpd/conf.d/
COPY --from=preparer out/idp-metadata.xml /etc/shibboleth/
COPY --from=preparer out/local.shibboleth2.xml /etc/shibboleth/

RUN chown -R shibd:shibd /etc/shibboleth/
RUN chown -R shibd:shibd /var/cache/shibboleth/