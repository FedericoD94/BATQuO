FROM python:3.7.8-slim-buster as base

# Leitha Proxy specifics
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VERSION=1.0.5

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Install requirements using poetry
WORKDIR /qaoa-pipeline
COPY poetry.lock pyproject.toml ./
RUN poetry install

COPY *.py .

# install the src as package as last thing to leverage Docker cache
RUN poetry install

ENTRYPOINT ["bash"]

