"""搜索模块的异常类型。

单独成文件是为了让 ``artifacts.py``（加载产物）与 ``service.py``（在线服务）
共用同一个异常而不产生循环 import；``service.py`` 会按契约重新导出它。
"""

from __future__ import annotations


class SearchNotReadyError(Exception):
    """搜索能力尚不可用：索引产物缺失 / 损坏，或请求的 mode 还没有实现。

    API 层应将其映射为 503 ``NOT_READY``。
    """
