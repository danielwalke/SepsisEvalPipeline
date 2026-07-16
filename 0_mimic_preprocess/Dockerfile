FROM rocker/r-ver:4.4.0

RUN apt-get update && apt-get install -y \
    zlib1g-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN install2.r --error \
    data.table

WORKDIR /app
RUN mkdir -p /app/input /app/output /app/extdata /app/features

COPY process_files.R /app/process_files.R
COPY extdata /app/extdata
COPY extract.py /app/extract.py
COPY entrypoint.sh /app/entrypoint.sh

RUN sed -i 's/\r$//' /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
