FROM ubuntu:26.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip libvulkan1 mesa-vulkan-drivers \
    libdrm2 libopenblas0 libgomp1 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/python
COPY requirements.txt /srv/requirements.txt
RUN /opt/python/bin/pip install --no-cache-dir --disable-pip-version-check -r /srv/requirements.txt
COPY docker/runtime/ /opt/crispasr/
ENV LD_LIBRARY_PATH=/opt/crispasr/lib
COPY server.py alignment_groups.py /srv/
WORKDIR /srv
RUN chmod 755 /opt/crispasr/bin/crispasr \
    && /opt/python/bin/python -m py_compile /srv/server.py /srv/alignment_groups.py
USER 1000:1000
ENTRYPOINT ["/opt/python/bin/python", "-m", "uvicorn"]
CMD ["server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
