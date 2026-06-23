# APLR聊天数据库生成器

独立Python程序，使用用户提供的微信、QQ等聊天记录，使用LLM批量提取事件，整理为满足 [本地回忆[APLR]](https://github.com/olozhika/astrbot_plugin_local_reminiscence) repo格式的数据库文件，实现网聊记忆数据化。适合用AI帮自己管理记忆或者人格切片、数字飞升等情形（这对吗）

作者已测试，可用好用【注意：生成的数据库结构将兼容APLR1.4.0版本更新后的数据库，APLR的该版本我还没写完，暂时不清楚接APLR1.3版本有没有什么bug(应该不严重)】


## 核心功能

- **智能总结**：利用大语言模型（LLM）将零散的对话整合为具有叙述性的事件。
- **并行处理**：支持多批次并行总结，自动根据CPU情况调整并发数。
- **自动重试**：API调用失败时自动重试，使用指数退避策略（2s、8s、16s、32s、64s），之后每次随机等待0.5-5秒，最多50次。重试时间这么长是为了避免被API公司夹死。
- **记忆提取**：自动识别并更新人物、地点、概念等"记忆节点"，支持增量更新和去重。
- **节点优化**：所有批次完成后，分批调用LLM优化节点，合并重复、去重别名。
- **增量更新**：当数据库已存在时，自动在已有数据基础上继续更新。
- **智能分批**：支持超长聊天记录，根据对话停顿（默认3小时）自动切分批次。
- **群聊脱水**：针对群聊文件，仅保留目标用户及其前后的上下文。
- **时间范围过滤**：支持指定起始和截至日期，自动剔除范围外及无自己发言的文件。
- **失败恢复**：失败批次自动保存，支持 `--retry` 重新运行。
- **多平台支持**：同时支持 QQ私聊、QQ群聊、微信私聊、微信群聊。
- **Astrbot 兼容**：生成的数据库结构与 [APLR](https://github.com/olozhika/astrbot_plugin_local_reminiscence) 插件完全一致。

## 工作流程

```
1. 加载配置 → 2. 日期过滤(删除无自己发言的文件) → 3. 群聊脱水
      ↓
4. 分批 → 5. 并行调用LLM总结 → 6. 写入数据库（后缀带有日期戳的db文件）
      ↓
7. 节点优化(合并去重) → 8. 完成（output_db）
```

详细流程：
1. **配置加载**：读取 `config.json` 中的 API 密钥、模型及处理参数。
2. **预处理**：日期范围过滤，同时删除无自己发言的聊天文件。
3. **群聊预处理**：对群聊文件进行脱水，仅保留目标用户相关的对话片段。
4. **智能分批**：按时间间隔或文件大小切分批次。
5. **并行总结**：多批次并行调用LLM，每个批次智能匹配相关节点作为背景。
6. **节点优化**：按类型分组，分批调用LLM优化节点描述。
7. **数据持久化**：将事件和节点存入 SQLite 数据库。

## 快速开始

### 1. 安装依赖

```bash
pip install pydantic openai jieba psutil
```

### 2. 准备数据

**聊天记录导出方式**：
- 微信：用 [WeFlow](https://github.com/hicccc77/WeFlow) 导出为 txt
- QQ私聊：旧版QQ用消息管理导出为 txt
- QQ群聊：用 [qq-chat-exporter](https://github.com/shuakami/qq-chat-exporter)

**目录结构**：

```
聊天记录/
├── QQ私聊/          # QQ私聊记录，文件名如: 用户名(QQ号).txt
├── QQ群聊/          # QQ群聊记录，文件名如: 群名.txt
└── 微信记录/        # 微信聊天记录
    ├── 私聊_用户名.txt
    └── 群聊_群名.txt    # 群聊文件名需包含"群聊"二字
```

**聊天记录格式**：

```text
2022-07-02 17:53:48 用户名
消息内容...
```

> ⚠️ 请务必在其他地方保存原始聊天记录，因为预处理会删除冗余文件。

### 3. 配置程序

编辑 `config.json` 文件：

```json
{
  "api": {
    "api_key": "您的API密钥",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o"
  },
  "prompts": {
    "ai_name": "您的名字",
    "system_prompt": "您的人格描述..."
  },
  "files": {
    "input_dir": "聊天记录",
    "output_db": "memory.db",
    "batch_size_kb": 40,
    "time_gap_hours": 3,
    "max_concurrent": 0,
    "max_retries": 50,
    "qq_id": "您的QQ号",
    "target_username": {
      "qq": "QQ私聊用户名",
      "wechat_private": "微信私聊显示名",
      "wechat_group": "微信群聊显示名"
    },
    "context_lines": 10,
    "max_nodes_context": 50,
    "optimize_batch_size": 20,
    "start_date": "2016-01-01",
    "end_date": "2019-12-31"
  }
}
```

**配置说明**：

| 参数 | 说明 |
|------|------|
| `input_dir` | 聊天记录根目录 |
| `output_db` | 输出数据库文件名 |
| `batch_size_kb` | 单次总结的最大字节数（默认40） |
| `time_gap_hours` | 判定对话中断的时间差（默认3小时） |
| `max_concurrent` | 并发数（0=自动，正整数=固定值） |
| `max_retries` | API失败最大重试次数（默认50） |
| `qq_id` | 您的QQ号（用于QQ群聊识别） |
| `target_username` | 各平台的用户名 |
| `context_lines` | 群聊脱水保留的上下文行数 |
| `max_nodes_context` | 每次传给LLM的最大节点数 |
| `optimize_batch_size` | 节点优化每批处理数 |
| `start_date` / `end_date` | 总结的时间范围 |

### 4. 运行

```bash
# 使用默认 config.json
python main.py  #请在配置config后再执行

# 可以指定配置文件（不需要写.json后缀）
python main.py myconfig

# 你也可以依次执行多个配置文件,每个配置文件执行前如果数据库db文件已存在会自动备份
python main.py config1 config2 config3

# 重新运行失败批次
python main.py --retry config1

# 如果你第二年有新的聊天记录需要补充，可以把output_db写成你已有的数据库文件直接追加并执行
python main.py
```

### 5. 自动确认模式

在 `main.py` 开头可以设置 `AUTO_CONFIRM` 开关：

```python
AUTO_CONFIRM = True   # 跳过确认，直接开始总结
AUTO_CONFIRM = False  # 需要输入 y 确认
```

## 注意事项

- **数据备份**：预处理会**直接修改并删除**原始文件中的冗余内容，请务必提前备份。
- **自动备份**：程序会在开始时备份数据库（`xxx_origin_时间戳.db`），节点优化前也会备份（`xxx_时间戳.db`）。
- **Token 消耗**：总结大量聊天记录会消耗较多 Token，程序启动时会显示预估。
- **API 兼容性**：支持所有兼容 OpenAI 接口标准的模型（如 DeepSeek, 智谱 AI, 月之暗面, MiMo等）。
- **失败恢复**：失败批次保存在 `logs/failed_batches/`，可用 `--retry` 重新运行。
- **日志文件**：每个配置文件有独立日志，保存在 `logs/` 目录。

## 数据库说明

生成的数据库包含以下核心表：
- `events`: 存储总结后的事件叙述、情绪、重要性等。
- `nodes`: 存储提取的实体/概念及其描述、别名、关联事件。
- `tags` & `event_tags`: 存储事件关联的标签。
- `daily_reflections`: 保持兼容性的空表。
- `event_relations`: 存储事件间的逻辑关系。

## 作者记忆大赏
下面这段就是作者用自己所有本科聊天记录提出出的数据库中，对于某一位老师的记忆
（自己的记忆里都是什么怪东西）
![自己的记忆里都是什么怪东西](./%E6%88%AA%E5%9B%BE%202026-06-23%2010-12-30.png)
