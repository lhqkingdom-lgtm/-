# P2：search_related_memories 工具设计与验收

## 目标

当 ReAct Agent 只看到有限 memory/topic 摘要，无法确认是否已有相关旧记忆时，允许它主动搜索长期记忆库，找到候选 memory，再决定是否继续读取详情、合并、更新、拆分、冲突记录或 noop。

## 工具边界

`search_related_memories` 是只读召回工具，不写库，也不授权修改已有 memory。

搜索结果只包含候选摘要和 ID。模型如果要修改已有记忆，仍必须继续调用 `read_memory_detail` 或 `read_topic_detail`，把目标 memory 放入 `modifiable_memory_ids` 后才能提交 `update_memory`、`merge_memory`、`mark_conflict` 或 `record_conflict_decision`。

## 搜索范围

当前第一版搜索以下字段：

- `memory_topics.title`
- `memory_topics.summary`
- `memories.summary`
- `memories.content`

返回结果会包含：

- `memory_id`
- `topic_id`
- `topic_title`
- `topic_summary`
- `memory_type`
- `summary`
- `matched_terms`
- `matched_fields`
- `score`
- `score_reason`
- `next_step`

## 参数

```json
{
  "query": "必填；来自当前 turn 证据或已有索引的真实关键词",
  "memory_type": "可选；限制 memory_type",
  "domain": "可选；限制领域 scope",
  "project_key": "可选；限制项目 scope",
  "limit": "可选；1-20，默认 8"
}
```

## 防幻觉设计

- 空 query 返回 `status=rejected` 和引导语，不抛无声异常。
- `limit` 在工具层限制为 1-20。
- 多关键词查询至少命中多个词，降低只命中泛词的噪声。
- 返回 `matched_fields` 和 `matched_terms`，让模型知道为什么命中。
- 返回 `next_step`，明确提示修改前必须读取详情。
- ReAct 校验层禁止未知参数，防止模型传入幻觉字段。

## 单元测试验收标准

已覆盖：

1. 中英文混合 query 能提取关键词。
2. 空 query 返回可解释 rejected。
3. limit 超大时被限制到 20。
4. 无结果时返回下一步搜索建议。
5. ReAct 校验接受合法 `search_related_memories` 请求。
6. ReAct 校验拒绝非法 `limit`。

## 接入测试验收标准

真实库测试应满足：

1. `Keil C51 编译错误` 能召回 Keil C51 编译失败/自检相关记忆。
2. `桩函数 固定返回` 能召回硬件输入桩函数相关记忆。
3. 搜索结果包含 `matched_fields`、`matched_terms` 和 `score_reason`。
4. 搜索结果不会直接进入 `modifiable_memory_ids`。
5. 如果模型要合并/更新/冲突处理，下一步必须调用 `read_memory_detail`。

## 后续优化

- P2.1：增加 `search_related_topics` 或把 topic-only 命中结果单独展示。
- P2.2：接入向量召回或 PostgreSQL 全文检索，解决同义表达召回不足。
- P2.3：记录搜索调用日志，用于评估模型是否过度搜索或漏搜。
- P2.4：结合冲突处理，把“同一语义槽位的新旧说法”搜索做得更准。
