# 结构1、2、3、4及各自搜索后的参数

四个独立JSON文件，每个同时包含节点类型、固定超参数、已训练参数和带权连边，可直接加载。使用的是完整F1→F3搜索后的最终F3参数，不是小规模搜索结果，不是随机初始参数，也不是串接组合。

| 文件 | 节点数 | 参数数 |
|---|---:|---:|
| structure_1.json | 5 | 26 |
| structure_2.json | 6 | 28 |
| structure_3.json | 8 | 31 |
| structure_4.json | 11 | 40 |

JSON格式为`{"blocks": [...], "edges": [...]}`。每个节点的`type`为算子类型，`kwargs`为固定超参数，`parameters`为搜索后的参数；`edges`每项是`[源节点索引, 目标节点索引, 权重]`，索引从0开始。Tournament参数连续值由算子取整，Exchange权重由算子归一化；保留原始搜索值以准确复现。

在E:/EC/deep中加载配置：

```python
from trained_structures.load_structures import load_config, load_structure

config1 = load_config(1)
blocks1, graph1 = load_structure(1, seed=0)
blocks2, graph2 = load_structure(2, seed=0)
blocks3, graph3 = load_structure(3, seed=0)
blocks4, graph4 = load_structure(4, seed=0)
```

也可以不用辅助文件，直接读取任意JSON并传给项目现有函数：

```python
import json
from pathlib import Path
from NeuroEA_GEA_torch.entropy_probe import build_structure
from NeuroEA_GEA_torch.utils.rng import NumpyTorchRNG

config = json.loads(Path('trained_structures/structure_1.json').read_text(encoding='utf-8'))
blocks, graph = build_structure(config, NumpyTorchRNG(0))
```

读取JSON本身只需Python标准库；构造和执行算子需本项目的NeuroEA_GEA_torch运行库及其依赖。四个JSON均已核对与`results/figure_structures/search_full/graph_00N/trained_structure.json`中的`structure_config`完全一致。
