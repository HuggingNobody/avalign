"""验证 ``import avalign`` 的纯 NumPy 路径不会牵连 PyTorch。

这个测试不依赖 torch，因此在主测试矩阵里也会运行；当环境里恰好装了 torch
时，它才真正发挥作用——确认惰性加载的 models/train 子包没有被顶层包拉起。
"""

import subprocess
import sys


def test_import_avalign_is_torch_free():
    code = (
        "import sys; import avalign; "
        "leaked = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.')); "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
