FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

COPY src ./vanity

RUN apt-get update -y \
    && apt-get install -y python3-pip vim

RUN pip install -r ./vanity/requirements.txt

RUN mkdir -p /etc/OpenCL/vendors && echo libnvidia-opencl.so.1 >> /etc/OpenCL/vendors/nvidia.icd