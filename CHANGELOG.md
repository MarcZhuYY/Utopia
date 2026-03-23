# Changelog

所有显著变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [0.2.0] - 2026-03-24

### 重大变更 (Breaking Changes)

- 重构Agent认知层，引入Pydantic v2严格校验
- 社交层使用NumPy张量重构，提升50倍性能
- 引擎层引入CQRS架构

### 新增功能

#### Phase 9: CQRS + Event Sourcing 架构
- **World Events** (`layer2_world/world_events.py`)
  - Pydantic v2事件模型，不可变设计
  - 6种领域事件: StanceChange, RelationshipCreate, OpinionCreate等

- **Event Buffer** (`layer2_world/world_event_buffer.py`)
  - asyncio.Queue无锁事件缓冲
  - 支持100+ agents并发

- **Neo4j Batch Mutator** (`layer2_world/neo4j_graph_mutator.py`)
  - UNWIND批处理写入器
  - 50 agents × events → 1 Neo4j transaction

- **Query Service** (`layer2_world/query_service.py`)
  - CQRS Query端，只读查询服务

- **Engine CQRS** (`layer5_engine/engine_cqrs.py`)
  - CQRS版本Tick生命周期
  - Query(只读) → Command(Agent决策) → Commit(批量写入)

#### Phase 10: Agent Persona 模板系统
- **Persona Models** (`layer3_cognition/agent_persona_models.py`)
  - Pydantic v2 Persona模型
  - 四大标准模板: RetailInvestor, QuantInstitution, Insider, MacroRegulator
  - Field校验: `ge=0.0, le=1.0` 严格边界
  - 复合校验: Regulator capital_weight强制10000.0

- **Agent Factory** (`layer3_cognition/agent_factory.py`)
  - 工厂模式创建Agent
  - 支持批量创建: `AgentFactory.create_batch()`
  - 支持参数覆盖

### 性能优化

- **L4社交层**: NumPy张量替代NetworkX遍历，O(n²) → O(n)
- **CQRS事件**: 无锁并行Agent执行
- **内存优化**: 三层记忆系统 (Hot/Warm/Cold)

### 测试

- 新增32个Phase 10测试
- 新增17个Phase 9测试
- **总计238个测试全部通过**

---

## [0.1.0] - 2026-03-20

### 初始版本

#### 核心功能
- 6层架构基础实现
- Agent基础认知系统
- NetworkX社交图
- 简单Tick引擎

#### 架构层次
- L1: 种子材料处理
- L2: 世界模型 (NetworkX)
- L3: 个体认知
- L4: 社交互动
- L5: 模拟引擎
- L6: 结果分析

---

## 版本对比

| 特性 | v0.1.0 | v0.2.0 |
|------|--------|--------|
| Agent并发 | 串行 | 100+ 并行 |
| 社交层 | NetworkX遍历 | NumPy张量 |
| 事件系统 | 无 | CQRS+Event Sourcing |
| Agent人格 | 简单模板 | 四大标准角色 |
| 数据库 | 无 | Neo4j ready |
| 测试数 | ~100 | 238 |

---

[0.2.0]: https://github.com/MarcZhuYY/Utopia/releases/tag/v0.2.0
[0.1.0]: https://github.com/MarcZhuYY/Utopia/releases/tag/v0.1.0
