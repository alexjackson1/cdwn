FROM verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3

WORKDIR /workspace

# Install dependencies
RUN git clone https://github.com/volcengine/verl . && pip3 install -e .

RUN git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git && \
  cp patches/megatron_v4.patch Megatron-LM/ && \
  cd Megatron-LM && git apply megatron_v4.patch && \
  pip3 install -e . && \
  cd ..

RUN pip3 install wandb

ENV PYTHONPATH="/workspace/Megatron-LM:${PYTHONPATH}"

COPY . .

# Run a command line
CMD ["bash"]