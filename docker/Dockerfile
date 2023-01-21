FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04
# FROM ubuntu:22.04
LABEL maintainer="Ryoya Kaneda"
LABEL MakeDay="2023/01/21"

ARG USERNAME=user
ARG GROUPNAME=group
ARG UID=1000
ARG GID=1000
ARG PYTHONVERSION="3.10.4"

ENV WORKSPACE /var/www
ENV ROOTHOME /root
ENV POETRY_HOME /opt/poetry
ENV RUST_HOME /usr/local/lib/rust
ENV RUSTUP_HOME ${RUST_HOME}/rustup
ENV CARGO_HOME ${RUST_HOME}/cargo
ENV PATH $POETRY_HOME/bin:$CARGO_HOME/bin:$PATH

WORKDIR $WORKSPACE
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git wget ssh curl locales gosu \
    build-essential libbz2-dev libdb-dev \
    libreadline-dev libffi-dev libgdbm-dev liblzma-dev \
    libncursesw5-dev libsqlite3-dev libssl-dev \
    zlib1g-dev uuid-dev tk-dev && \
    apt-get -y clean && rm -rf /var/lib/apt/lists/*

# set Timezone and Language
ENV TZ=Asia/Tokyo
RUN locale-gen ja_JP.UTF-8 && ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo $TZ > /etc/timezone
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8

# Python3 install
RUN wget https://www.python.org/ftp/python/${PYTHONVERSION}/Python-${PYTHONVERSION}.tar.xz -O - | tar xJf - && \
    cd Python-${PYTHONVERSION} && ./configure && make && make install && \
    cd ../ && rm -rf Python-${PYTHONVERSION} && \
    pip3 install --upgrade pip && pip3 install setuptools entrypoints

# Rust and cargo-make install
RUN mkdir $RUST_HOME && chmod 0755 $RUST_HOME && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > ${RUST_HOME}/rustup.sh && \
    chmod +x ${RUST_HOME}/rustup.sh && ${RUST_HOME}/rustup.sh -y --default-toolchain nightly --no-modify-path && \
    cargo install --force cargo-make

# poetry initalize
COPY pyproject.toml $WORKSPACE
COPY src/ ${WORKSPACE}/src/
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false && \
    poetry install --no-cache --with jupyter,torch-cu117 && \
    jupyter contrib nbextension install && jupyter nbextensions_configurator enable

# add user and set user
RUN groupadd -g $GID $GROUPNAME && useradd -m -s /bin/bash -u $UID -g $GID $USERNAME
USER $USERNAME

CMD ["python3"]
