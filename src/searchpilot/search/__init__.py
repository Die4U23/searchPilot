"""搜索核心：归一化与分词、倒排索引、BM25、RRF、产物持久化与在线服务。

对外入口（见 docs/dev/build-plan.md 第 2 节）：

- ``searchpilot.search.service.build_search_service(artifact_dir)``
- ``searchpilot.search.service.InMemorySearchService(documents)``
- ``searchpilot.search.service.SearchNotReadyError``
"""

from __future__ import annotations

from searchpilot.search.errors import SearchNotReadyError

__all__ = ["SearchNotReadyError"]
