# Utopia v0.2.0

**高保真多智能体模拟引擎**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-238%20passing-brightgreen)](tests/)

Utopia是一个开源的多智能体模拟系统，专门设计用于通过真实的智能体行为建模集体智能。它模拟个体如何感知信息、形成信念、进行社交互动，以及群体动态如何涌现。

---

## 核心特性

- **6层架构**: 从种子材料处理到结果分析的模块化设计
- **认知智能体**: 记忆系统、信念网络、决策引擎
- **社交动力学**: 信息传播伴随认知扭曲、极化检测
- **Agent人格系统**: 四大标准模板（散户、量化机构、内幕者、国家队）
- **CQRS + Event Sourcing**: 支持100+智能体并发执行
- **可扩展**: 易于添加新领域（金融、政治等）
- **预测引擎**: 可集成为量化交易的预测引擎

---

## 架构

```
┌─────────────────────────────────────────┐
│  L6: Result Analysis                    │
│      指标收集 / 预测 / 报告生成           │
├─────────────────────────────────────────┤
│  L5: Simulation Engine                  │
│      Tick循环 / 调度 / CQRS事件          │
├─────────────────────────────────────────┤
│  L4: Social Interaction                 │
│      关系网络 / 信息传播张量              │
├─────────────────────────────────────────┤
│  L3: Individual Cognition               │
│      记忆 / 信念 / 决策 / Persona         │
├─────────────────────────────────────────┤
│  L2: World Model                        │
│      知识图谱 / 事件溯源 / Neo4j         │
├─────────────────────────────────────────┤
│  L1: Seed Material Processing           │
│      实体抽取 / 解析                      │
└─────────────────────────────────────────┘
```

---

## 安装

```bash
pip install utopia-sim
```

或从源码安装:

```bash
git clone https://github.com/MarcZhuYY/Utopia.git
cd Utopia
pip install -e ".[all]"
```

---

## 快速开始

### 基础示例

```python
from utopia.layer3_cognition.agent_factory import AgentFactory
from utopia.layer5_engine.world_state_buffer import WorldStateBuffer

# 创建不同类型的Agent
retail = AgentFactory.create_agent("retail", "RETAIL_001")
quant = AgentFactory.create_agent("quant", "QUANT_001")
regulator = AgentFactory.create_agent("regulator", "CSRC_001")

print(retail.to_system_prompt())
```

### 批量创建智能体

```python
# 创建70个散户 + 15个量化机构 + 5个内幕者
agents = AgentFactory.create_batch({
    "retail": 70,
    "quant": 15,
    "insider": 5,
})

# 或者使用参数覆盖
custom_agents = AgentFactory.create_batch({
    "retail": (50, {"neuroticism": 0.95}),  # 更恐慌的散户
    "quant": (10, {"openness": 0.05}),       # 更保守的量化
})
```

---

## Agent人格模板系统

| 角色 | 开放度 | 神经质 | 尽责性 | 影响力 | 资金权重 |
|------|--------|--------|--------|--------|----------|
| **RetailInvestor** (散户) | 0.8 | 0.9 | 0.2 | 0.1 | 1.0 |
| **QuantInstitution** (量化) | 0.1 | 0.0 | 1.0 | 0.5 | 100.0 |
| **Insider** (内幕) | 0.3 | 0.2 | 0.8 | 0.8 | 50.0 |
| **MacroRegulator** (国家队) | 0.1 | 0.0 | 1.0 | 1.0 | **10000.0** |

### 人格参数对行为的影响

- **Openness (开放度)**: 影响贝叶斯信念更新吸收率
- **Neuroticism (神经质)**: 影响认知失调时的信心惩罚
- **Conscientiousness (尽责性)**: 影响记忆衰减率
- **Influence Weight (影响力)**: 影响L4信息传播权重

```python
from utopia.layer3_cognition.agent_persona_models import RetailInvestorPersona

# 获取记忆衰减率 (散户: 0.08, 量化: 0.0)
decay_rate = retail.get_memory_decay_rate()

# 获取贝叶斯更新率 (散户: 0.8, 量化: 0.1)
update_rate = retail.get_bayesian_update_rate()

# 计算认知失调惩罚
penalty = retail.get_confidence_penalty(dissonance_level=0.5)
```

---

## 领域特定扩展

### 金融市场模拟

```python
from utopia.core.config import SimulationConfig
from utopia.layer5_engine.engine import SimulationEngine

config = SimulationConfig(
    agent_count=100,
    max_ticks=100,
    domain="financial",
)

engine = SimulationEngine(config)
engine.initialize(seed_material)
result = engine.run()
```

### 政治分析模拟

```python
config = SimulationConfig(
    agent_count=50,
    max_ticks=50,
    domain="political",
)
```

---

## API 参考

### AgentFactory

```python
AgentFactory.create_agent(
    role: str,           # "retail" | "quant" | "insider" | "regulator"
    agent_id: str,      # 唯一标识符
    **kwargs            # 可选参数覆盖
) -> BaseAgentPersona

AgentFactory.create_batch(
    counts: dict        # {role: count} 或 {role: (count, kwargs)}
) -> list[BaseAgentPersona]
```

### 信念系统 (Beliefs v2)

```python
from utopia.layer3_cognition.beliefs_v2 import BayesianBeliefSystem, StanceState

# 初始化立场
stance = StanceState(
    topic_id="MARKET",
    position=0.5,      # -1.0 (反对) 到 1.0 (支持)
    confidence=0.7,    # 0.0 到 1.0
)

# 贝叶斯更新
belief_system = BayesianBeliefSystem(agent_openness=0.8)
new_stance = belief_system.update_stance(
    current=stance,
    evidence_position=0.8,
    evidence_confidence=0.6,
)
```

### CQRS 事件溯源

```python
from utopia.layer2_world.world_events import StanceChangeEvent
from utopia.layer2_world.world_event_buffer import WorldEventBuffer

# 创建不可变事件
event = StanceChangeEvent(
    agent_id="RETAIL_001",
    topic_id="MARKET",
    new_position=0.8,
    reason="positive_news",
)

# 缓冲事件
buffer = WorldEventBuffer()
buffer.add_event(event)
```

---

## 设计原则

1. **真实性优先于规模** - 100个真实智能体 > 10,000个简单智能体
2. **认知层次分离** - 世界模型、个体认知、社交互动解耦
3. **可验证性** - 每个智能体的推理可追溯

---

## 路线图

### v0.2.0 (当前)
- ✅ Phase 9: CQRS + Event Sourcing 架构
- ✅ Phase 10: Agent Persona 模板系统
- ✅ 238个测试全部通过

### v0.3.0 (计划中)
- [ ] LLM集成 (MiniMax API)
- [ ] 市场情绪指数模块
- [ ] 波动率预测

### v1.0.0 (未来)
- [ ] Neo4j图数据库集成
- [ ] 向量存储集成 (Milvus/Qdrant)
- [ ] React + D3.js 可视化仪表板
- [ ] REST API for 交易系统集成
- [ ] 金融市场专用智能体和规则

---

## 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目。

---

## 许可

MIT License - 详见 [LICENSE](LICENSE) 文件。

---

## 作者

**Zhu Yi** - [@MarcZhuYY](https://github.com/MarcZhuYY)

---

## 致谢

- 灵感来源于Alpha-A量化交易系统的智能体需求
- 架构设计受CQRS和Event Sourcing模式启发
