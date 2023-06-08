# Set the base image
ARG BASEIMAGE
FROM ${BASEIMAGE}

# Install python3-venv for the virtual environment
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv

# Create a virtual environment
ENV VIRTUAL_ENV=/home/scrapalot/scrapalot-chat/.venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set work directory
WORKDIR /home/scrapalot/scrapalot-chat

# Copy required files
COPY scripts ./scripts
COPY ["requirements_linux.txt", "scrapalot_ingest.py", "scrapalot_main.py", "scrapalot_main_api_run.py", "scrapalot_main_web.py", "scrapalot_logs.py", "./"]

# Install dependencies
RUN pip3 install -r requirements_linux

# Expose port
EXPOSE 8080 8501
