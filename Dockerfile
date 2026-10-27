FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 依赖层：只依赖 pyproject.toml / README.md / src，改动业务代码才会重装
COPY pyproject.toml README.md ./
COPY src ./src
# 向量额外依赖含 torch。先装 CPU 轮子，避免 .[vector] 拉到 CUDA 包。
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install ".[vector,agent]"

# 非 root 运行；data/ 与 artifacts/ 由 compose 挂载进来。
# 句向量模型在构建时下载到 searchpilot 家目录，运行时不再访问网络。
RUN useradd --system --create-home --uid 10001 searchpilot \
    && mkdir -p /app/data /app/artifacts /home/searchpilot/.cache \
    && chown -R searchpilot:searchpilot /app/data /app/artifacts /home/searchpilot
USER searchpilot
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1

# 脚本和迁移变更不应使固定模型下载层失效。
COPY db ./db
COPY scripts ./scripts

ENV SEARCHPILOT_DATA_DIR=/app/data \
    SEARCHPILOT_ARTIFACT_DIR=/app/artifacts \
    SEARCHPILOT_VECTOR_WARMUP=true \
    SEARCHPILOT_DISABLE_THP=true \
    MALLOC_ARENA_MAX=2

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=240s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"

CMD ["python", "scripts/docker_entrypoint.py"]
