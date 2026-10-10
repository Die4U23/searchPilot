FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 依赖层：只依赖 pyproject.toml / README.md / src，改动业务代码才会重装
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# 迁移与运维脚本（python scripts/migrate_database.py 可在容器内执行）
COPY db ./db
COPY scripts ./scripts

# 非 root 运行；data/ 与 artifacts/ 由 compose 挂载进来
RUN useradd --system --create-home --uid 10001 searchpilot \
    && mkdir -p /app/data /app/artifacts \
    && chown -R searchpilot:searchpilot /app/data /app/artifacts
USER searchpilot

ENV SEARCHPILOT_DATA_DIR=/app/data \
    SEARCHPILOT_ARTIFACT_DIR=/app/artifacts

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"

CMD ["python", "scripts/docker_entrypoint.py"]
