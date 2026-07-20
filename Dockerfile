FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL project="poi-recommendation"
LABEL framework="pytorch"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# 基础镜像已经包含 GPU 版 PyTorch，安装其余项目依赖时跳过 torch。
COPY requirements.txt /tmp/requirements.txt
RUN grep -Ev '^torch([<>=!~].*)?$' /tmp/requirements.txt \
        > /tmp/requirements-no-torch.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-no-torch.txt \
    && python -c "import torch; assert torch.__version__.startswith('2.5.1')"

COPY . /workspace

RUN mkdir -p \
    /workspace/results/checkpoints \
    /workspace/results/metrics \
    /workspace/results/logs

# auto 在 GPU 平台使用 CUDA，在无 GPU 环境自动回退到 CPU。
CMD ["python", "experiments/run_gru.py", "--device", "auto", "--num-workers", "2"]
