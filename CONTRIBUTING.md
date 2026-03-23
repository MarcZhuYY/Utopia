# 贡献指南

感谢您对Utopia项目的兴趣！我们欢迎各种形式的贡献，包括代码、文档、测试、问题报告等。

---

## 如何贡献

### 报告问题

如果您发现了bug或有功能建议，请通过GitHub Issues提交：

1. 检查是否已有相关问题
2. 使用对应的issue模板
3. 提供详细的重现步骤（对于bug）
4. 标注相关标签

### 提交代码

#### 1. Fork和克隆

```bash
# Fork仓库后克隆
git clone https://github.com/YOUR_USERNAME/Utopia.git
cd Utopia
```

#### 2. 创建开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -e ".[all]"
```

#### 3. 创建分支

```bash
# 从master创建特性分支
git checkout -b feature/your-feature-name

# 或修复分支
git checkout -b fix/issue-description
```

#### 4. 开发和测试

```bash
# 运行测试
pytest tests/ -v

# 代码格式化
ruff check . --fix
ruff format .

# 类型检查（如果有mypy）
# mypy utopia/
```

#### 5. 提交更改

```bash
git add .
git commit -m "feat: 添加新特性"
git push origin feature/your-feature-name
```

#### 6. 创建Pull Request

- 在GitHub上创建PR
- 描述更改内容和动机
- 确保所有CI检查通过
- 等待代码审查

---

## 代码规范

### Python代码风格

- 遵循PEP 8规范
- 使用Ruff进行代码检查和格式化
- 行长度限制: 100字符
- Python版本: 3.10+

### 文档规范

- 所有公共API必须有docstring
- 使用Google风格docstring
- 复杂逻辑需要注释说明

```python
def example_function(param1: str, param2: int) -> bool:
    """简短描述。

    详细描述（如果需要）。

    Args:
        param1: 参数1的描述
        param2: 参数2的描述

    Returns:
        返回值的描述

    Raises:
        ValueError: 何时抛出此异常
    """
    pass
```

### 测试规范

- 所有新功能必须有测试
- 使用pytest框架
- 测试命名: `test_*.py`
- 测试函数命名: `test_function_name`

```python
def test_new_feature():
    """测试新特性。"""
    result = new_feature()
    assert result == expected
```

---

## 架构贡献指南

### 添加新的Layer

如果需要在架构中添加新的层：

1. 在 `utopia/` 下创建新目录 `layerX_name/`
2. 创建 `__init__.py` 暴露公共API
3. 在 `tests/` 添加对应的测试文件
4. 更新文档

### 添加新的Agent角色

1. 在 `agent_persona_models.py` 中继承 `BaseAgentPersona`
2. 设置默认参数值
3. 在 `AgentFactory._ creators` 中注册
4. 添加测试用例

```python
class NewRolePersona(BaseAgentPersona):
    role: AgentRole = Field(default=AgentRole.NEW_ROLE, frozen=True)
    openness: float = Field(default=0.5, ge=0.0, le=1.0)
    # ... 其他参数
```

### 添加新的事件类型

1. 在 `world_events.py` 中继承 `BaseWorldEvent`
2. 设置 `event_type` 和 `event_version`
3. 使用 `frozen=True` 确保不可变性
4. 添加序列化/反序列化测试

---

## 提交信息规范

使用[约定式提交](https://www.conventionalcommits.org/zh-hans/v1.0.0/)：

```
<type>[(scope)]: <subject>

[body]

[footer]
```

### 类型

- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具/依赖更新

### 示例

```
feat(layer3): 添加新的Agent人格参数

添加risk_tolerance参数，影响投资决策。

git commit -m "feat(layer3): 添加新的Agent人格参数

添加risk_tolerance参数，影响投资决策。"
```

---

## 审查流程

### PR审查检查清单

- [ ] 代码符合项目风格
- [ ] 所有测试通过
- [ ] 新增代码有测试覆盖
- [ ] 文档已更新
- [ ] 没有引入不必要的依赖
- [ ] 性能影响已评估

### 审查者指南

- 保持建设性和尊重
- 解释"为什么"而不仅仅是"什么"
- 区分"必须修改"和"建议"
- 及时响应

---

## 发布流程

### 版本号规则

遵循[语义化版本](https://semver.org/lang/zh-CN/)：

- **MAJOR**: 不兼容的API修改
- **MINOR**: 向下兼容的功能新增
- **PATCH**: 向下兼容的问题修复

### 发布步骤

1. 更新 `pyproject.toml` 中的版本号
2. 更新 `CHANGELOG.md`
3. 创建Git标签
4. 推送标签触发发布

```bash
git tag -a v0.3.0 -m "Release v0.3.0"
git push origin v0.3.0
```

---

## 社区

### 沟通渠道

- GitHub Issues: Bug报告和功能请求
- GitHub Discussions: 一般讨论和问答

### 行为准则

- 尊重所有贡献者
- 欢迎新手提问
- 专注于建设性讨论
- 遵守开源社区规范

---

## 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

---

## 致谢

感谢所有为Utopia做出贡献的开发者！

特别感谢：
- Alpha-A量化交易系统提供的灵感
- 开源社区的工具和库
