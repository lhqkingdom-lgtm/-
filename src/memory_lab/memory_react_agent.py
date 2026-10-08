"""ReAct-style memory maintenance agent.

This module replaces the fixed one-shot writer flow with a small, bounded
reason/act loop:

turn evidence + memory indexes -> model asks for read tools or final changes
-> local code validates/executes read tools -> model emits memory-changes-v1
-> existing deterministic validator/committer handles writes.

The model never receives a write tool.  Tool calls are JSON requests that must
pass local validation, and final writes still go through ``memory_changes``.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_CONFIG
from .llm_client import OpenAICompatibleJsonClient
from .memory_agent import (
    DEFAULT_AGENT_RUN_DIR,
    _context_stats,
    _safe_run_id,
    _step,
    build_memory_agent_input,
)
from .memory_changes import (
    ALLOWED_MEMORY_TYPES,
    MEMORY_CHANGES_SCHEMA_VERSION,
    MemoryChangesValidationError,
    commit_memory_changes,
    validate_memory_changes,
)
from .memory_chain_state import (
    CLOSED_STATES,
    MemoryChainStateValidationError,
    validate_chain_operations,
    validate_chain_state,
)
from .pg_store import (
    DEFAULT_PG_CONFIG,
    expand_pointer,
    get_current_turn_evidence_summary,
    list_open_memory_chains,
    list_previous_session_turns,
    list_memory_conflicts,
    list_topic_memories,
    read_memory_detail,
    read_memory_history,
    read_session_turn_evidence,
    read_topic_detail,
    save_memory_chain_state,
    search_related_memories,
    trace_memory_relations,
)


MEMORY_REACT_SCHEMA_VERSION = "memory-react-v1"
DEFAULT_REACT_RUN_DIR = Path(__file__).resolve().parents[2] / "reports" / "memory-react-runs"
ALLOWED_TOOLS = {
    "read_topic_detail",
    "read_memory_detail",
    "expand_pointer",
    "search_related_memories",
    "get_current_turn_evidence_summary",
    "trace_memory_relations",
    "list_topic_memories",
    "read_memory_history",
    "list_memory_conflicts",
    "list_open_memory_chains",
    "list_previous_session_turns",
    "read_session_turn_evidence",
}
CONFLICT_RECORD_STATUSES = {
    "needs_review",
    "resolved_by_update",
    "resolved_by_merge",
    "discarded_as_stale",
    "carried_forward",
}


class MemoryReactValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


REACT_SYSTEM_PROMPT = """你是 Memory Lab 的 ReAct 长期记忆整理 Agent，不是普通摘要器。
你的职责是把一轮 CP 对话整理成“长期记忆维护计划”，让未来相似任务少问、少错、少返工。

你只能输出一个严格 JSON 对象，不要输出 Markdown、解释、代码块或额外文本。

====================
一、你拿到的输入
====================

1. 当前 turn 的三层证据池：
- core：本轮最核心的用户反馈、结论、验证、关键要求。
- candidate：可能有价值的工具结果、过程摘要、工程上下文。
- pointer：可回查原始消息、工具输出、代码片段的指针。

2. 已有长期记忆索引：
- topic_index：主题摘要，只能用于发现候选主题。
- memory_index：记忆摘要，只能用于发现相似/重复/冲突候选。

3. observations：
- search_related_memories / read_memory_detail / read_topic_detail / list_topic_memories / read_memory_history / list_memory_conflicts / expand_pointer / get_current_turn_evidence_summary / list_open_memory_chains / list_previous_session_turns / read_session_turn_evidence 返回的详情。
- search_related_memories 只用于发现候选 memory，不代表已经读取正文，也不授权修改。
- list_topic_memories 只列出某个主题下的 memory 摘要，用于判断是否需要继续 read_memory_detail。
- read_memory_history 用于查看某条 memory 的版本历史，不授权修改。
- list_memory_conflicts 用于查看已有冲突记录，避免重复记录冲突或判断是否需要补充冲突证据；默认只返回 needs_review/carried_forward 这类活跃风险，只有审计历史时才请求 include_resolved=true。
- get_current_turn_evidence_summary 只用于回看当前 turn 的证据池，帮助确认 evidence_id 和本轮事实，不授权修改任何 memory。
- list_open_memory_chains 只返回同会话未闭环技术链及关联 turn 摘要；它能发现相关历史 turn，但不授权引用摘要中的 evidence_id。
- 任何 compilation/runtime 的最终判断（包括 noop + chain_state）前，都必须先调用一次 list_open_memory_chains。只有当前现象/失败、根因或修复与开放链属于同一故障时，才继续既有链并原样复制工具返回的 chain_id；仅仅同属 runtime/compilation、同会话或同一条多问题 turn 不算命中。根因或验证对象不同就省略 chain_id，形成独立链，绝不能用一个问题的用户确认关闭另一个问题的开放链。禁止编造 chain_id。
- list_previous_session_turns 只发现同会话更早 turn 的摘要，不授权引用摘要里的 evidence_id。
- 只有继续调用 read_session_turn_evidence 读取历史 turn 详情后，该 turn 和返回的 evidence_id 才进入跨 turn 证据授权集合。
- trace_memory_relations 用于查看某条 memory 的版本历史和 supersedes/merges/conflicts_with 关系，不授权修改。
- 只有 observations 中已经读到正文的 memory，才算可安全修改。

4. memory_access_state：
- summary_only_memory_ids：只看过摘要，不能直接 update/merge/conflict。
- observed_memory_ids：已经读取过正文。
- modifiable_memory_ids：允许在 final_changes 中 update/merge/merge_existing/retire/move_memory_topic/mark_conflict/record_conflict_decision。

====================
二、长期记忆沉淀标准
====================

长期记忆只保存以后能减少重复问答、错误或返工的信息，memory_type 只能是：
- user_preference：用户明确表达的协作习惯、长期规则或关键纠正。
- compile_failure_experience：编译、链接、构建或烧录前失败的症状、机制、修复与验证。
- runtime_failure_experience：烧录/运行现象、代码或硬件配置根因、解决方法与验证。
没有这三类价值时输出 noop；禁止发明近义类型。
已有库中可能出现旧类型 workflow、compile_failure_pattern、runtime_failure_pattern；它们只用于兼容读取和更新，分别按 user_preference、compile_failure_experience、runtime_failure_experience 理解。新建 memory 仍只能使用上面三种新类型。

不要按整轮主题或是否出现具体参数直接决定存不存。先逐项完成：
1. 事实提取：分别检查 user、assistant、tool 中的独立结论；不能用用户问题或整轮主题的价值覆盖其他证据，不把推测写成事实。
2. 机制抽象：去掉项目名、路径、具体数值后，写出仍成立的因果机制或协作规则。
3. 复用判断：抽象结果必须能指导未来相似任务；否则只是当前项目事实。
4. 证据准入：检查对应 memory_type 所需的证据和验证状态。

沉淀前必须先过三道门槛：

1. 长期价值门槛
- 只能回答“当前项目这次怎么做”的信息默认 noop。
- 具体寄存器、通道、工具版本或配置值本身不是记忆；但由工具授权等级、能力上限、工程识别或实际编译范围造成的可重复失败、结果失真及排查方法，属于可复用的工具链机制，不能仅因发生在当前环境而按项目参数过滤。

2. 类型必备证据门槛
- user_preference：必须来自用户明确表达。未来取向、跨 turn 重复，或明确建立会持续影响后续迭代的协作机制是强证据；一次表达若包含可复用的“触发条件—做法—目的”，可低置信度沉淀。单次对象明确的临时命令仍然 noop。
- compile_failure_experience：必须有明确失败/限制、证据支持的原因、已执行的修复，以及对“原始失败动作”重新执行成功的验证。未达到 compile_verified 时只保留 turn 证据并输出 noop + chain_state；“编译通过”本身不是长期经验，除非原始失败就发生在该编译/链接/构建步骤，且同类命令重跑成功。若原始错误发生在烧录/下载/编程阶段，仅有上游编译 0 Error 或生成 HEX 不能证明该错误消失；应保持 fix_applied 并把“原烧录/下载动作重试成功”列入 missing_evidence。
- 编译工具返回的“可能含义”、通用错误说明或错误码释义不是已证实根因。若同一环境随后用另一入口/命令成功构建，不能直接归因于许可证、安装或环境损坏；必须先定位两种调用的关键差异。原因仍不明确时保留 failure_observed 未闭环链并 noop，不得把“换个入口后成功”包装成完整失败经验。
- runtime_failure_experience：必须有用户反馈或工具确认的运行现象，以及定位原因和解决方向；只有现象时 request_more_evidence 或 noop。
- runtime_failure_experience 只有在用户测试/重新烧录/运行后明确确认正常，达到 user_verified，才能写入正式 memory。用户明确表示“我测试了，功能已实现/问题消失/现在正常”即属于 user_verified，不要求机械地逐字出现“烧录”；代码修好或编译通过都不能代替用户验证。

3. 当前项目参数过滤门槛
- 阈值、IO、PWM、定时、路径、产品名、功能开关、文件清单和一次性交付要求默认只作证据背景。
- 若去掉这些细节后不能留下独立、可执行的机制或协作规则，输出 noop。
- 具体细节只能限定适用范围，不能成为 summary 或 stable_key 的主体。

需求文档存当前项目规格；长期记忆存跨相似任务仍有用的规则。没有证据支持的猜测应 noop；有证据支持但尚未完成技术闭环的原因或修复方向仍只保留在 turn 证据中，通过 chain_state 说明当前状态和缺口，不创建或更新正式技术 memory。不要沉淀完整日志、完整工具输出或敏感信息。

单轮多条记忆自检：
- 一轮 turn 可以产生多条 memory，但每条必须有不同的核心结论、根因或用户偏好。
- 如果只是同一根因的多个表现，合成一条。
- 如果只是一个大问题的排查过程，不要把每个中间步骤都拆成 memory。
- 不要把“症状、原因、修复、验证”拆成多条 memory；它们应组成同一条经验链。
- 同一轮里多个运行现象如果来自同一个模块、同一状态机、同一硬件配置误解或同一类初始化问题，默认合成一条“故障家族”记忆。
- 只有当未来检索入口明显不同、根因不同、修复方法不同，且合并后会让记忆难以复用时，才拆成多条 memory。
- 如果同一轮要新增超过 2 条 memory，必须在总 reason 中逐条说明它们为什么有不同的未来检索入口和不同根因；否则优先合并或 noop 低价值项。

每条 memory 的质量底线：
- summary 必须表达可复用规则，而不是当前项目现象标题。
- content 至少包含：适用范围/触发条件、现象或失败症状、根因、处理原则或修复方向、验证状态。
- 如果缺少根因、修复方向或对应的最终验证，不得创建或更新正式技术 memory；输出 noop + chain_state，并明确缺少的证据。
- 不要只写“某项目出现了某问题”；要写成“什么条件下容易出现什么问题，为什么，之后怎么避免/排查”。
- 具体引脚、阈值、寄存器值、路径、文件名只能作为例子或证据背景，不能成为 summary 的主体，也不能作为 stable_key 的主要语义。

====================
三、动作选择标准
====================

create_memory：新增记忆
- 本轮信息有长期价值。
- 没有已有 memory 能覆盖。
- 可以形成清晰 stable_key。
- 有足够 evidence_refs。
- 只负责创建 memory；如果使用已有主题，必须使用 topic_id。
- 只有确实需要创建全新主题时才携带 topic 对象；create_memory 不能修改已有 topic。
- stable_key 和 topic_key 是创建时的路由标识，本阶段不可修改；不要为了“修正 key 语义”而 update_memory 或 update_topic。
- 如果新信息虽然属于同一个大主题，但形成了新的独立规则，应 create_memory，不要为了减少条目数强行 merge 到宽泛旧 memory。
- 原子化指“一条 memory 一个可复用核心结论”，不是“一条现象一条 memory”。同一故障链条中的症状、根因、修复、验证必须放在同一条 memory。
- 一轮 turn 同时包含用户偏好、编译失败、运行现象三类长期价值时，可以拆成多条；但同一类别内默认先合并同根因/同修复策略的内容。
例：第一次发现“硬件输入桩函数固定返回值会导致主流程跑通但输入恒定，下游状态机无响应”。

merge_memory：合并证据
- 已有 memory 是同一模式。
- 本轮只是新增证据、补充例子、增强置信度。
- 不改变原结论。
- 判断“同一模式”时要严格：新证据必须能用目标 memory 的同一句核心结论解释；如果需要新增另一个结论、另一个协作偏好、另一个失败原因或另一个修复方法，就不要 merge，应 create_memory 或 update_memory。
- 只表示“把当前 turn 证据合并到一个已有 memory”，不是多个已有 memory 合并。
- 不能借 merge_memory 修改 topic；如果 topic 标题/概要需要变化，单独输出 update_topic。
- merge_memory 不是重新写一篇报告：优先保留已有 memory 的核心 summary/content，只添加本轮真正新增的事实、验证状态、适用条件或反例。
- 如果本轮只是重复证明同一结论，content 只需补一句“新增证据：……”，不要复制完整日志、完整执行流、完整错误列表或大段会话。
- 如果目标 memory 已经很宽泛或内容已经较长，默认不要继续扩写；除非本轮证据直接增强同一句核心结论，否则应新建更小的 memory，或 noop。
- 不要把用户习惯塞进工程流程记忆，也不要把编译失败经验塞进运行现象记忆；这些只是在同一主题下相关，不等于同一条 memory。
- 如果本轮只提供未验证现象或助手推测，不能创建或更新正式技术 memory，也不能提高已有 memory 的 confidence；只记录 chain_state 和缺少的验证证据。
- 如果目标 memory 已包含历史“已验证”结论，而本轮只是未验证新现象，不要把未验证线索合并进正式 memory；输出 noop + chain_state，原有已验证结论保持不变。
- 只有出现用户确认、编译/烧录/运行验证、明确工具结果时，才可以把“解决有效/已验证”写入 content，并适度提高 confidence。
例：另一个项目也出现采样桩固定值导致状态不变化。

merge_existing_memories：合并多条已有记忆
- 多条 active memory 语义重复、同方向，继续分开会增加召回噪声或维护成本。
- 指定一个 target memory 作为合并后的当前有效记忆；source_memory_ids 中的记忆会标记为 merged。
- 必须给出合并后的完整 memory summary/content/confidence。
- 会记录 source → target 的 merges relation。
- 如果 source memory 上有 active conflict，必须提供 conflict_disposition，说明是 resolved_by_merge、discarded_as_stale 还是 carried_forward。
- 不能修改 topic；如果合并后 topic 语义需要变化，必须另行 update_topic。

update_memory：更新已有记忆
- 已有 memory 方向正确。
- 本轮提供了更准确、更通用、更完整的表述。
- 需要改写 summary/content。
- 只负责更新 memory 当前内容和 revision，不能修改 topic。
- 不能修改 stable_key；如果只是 stable_key 旧但 summary/content 已准确，通常 noop，不要为了 key 发起 update_memory。
例：原记忆只写 ADC 桩导致风扇不转，本轮抽象为“硬件输入桩函数导致下游控制逻辑处理假数据”。

retire_memory：退役不再适用的记忆
- 已有 active memory 已经失效、重复到不再需要单独保留，或被后续整理明确废弃。
- 只负责把 memory 标记为 retired 并写 revision；不能修改 topic，不能合并其他 memory，不能删除数据。
- 如果目标 memory 上有 active conflict，必须提供 conflict_disposition，说明这些冲突是 discarded_as_stale 还是 carried_forward。
- 适合“这条旧记忆不应再作为当前有效记忆参与后续判断”的场景。

move_memory_topic：移动记忆所属主题
- memory 内容正确，但归属主题错误或后续整理发现更适合放到另一个 active topic。
- 只负责把 memory 从 from_topic_id 移到 to_topic_id，并写 revision；不能修改 memory summary/content/confidence，也不能修改 topic title/summary。
- 必须提供 from_topic_id 和 to_topic_id；from_topic_id 必须匹配当前 memory 所属主题。
- 如果移动后 source/target topic 标题或概要不准确，必须另行输出 update_topic。

mark_conflict：标记冲突
- 同 scope、同条件、同对象下结论相反。
- 无法用项目、工具链、芯片、版本、环境差异解释。
- 只负责把 memory 标记为 conflict 并写 revision，不负责记录冲突决策。
- 如果需要说明旧说法、新说法和处理判断，必须另行输出 record_conflict_decision。
工程经验不要简单按时间覆盖；能拆条件就拆，不能解释才冲突。

update_topic：维护主题元信息
- 当已有 memory 的 summary/content 已被更新，但 topic 标题或概要仍停留在旧语义时使用。
- 只用于改主题 title/summary，不用于改记忆正文。
- update_topic 是唯一允许修改 topic.title/topic.summary 的动作。
- 不能修改 topic_key；如果只是 topic_key 旧但 title/summary 已准确，通常 noop，不要为了 key 发起 update_topic。
例：memory 已更新为“用户偏好英文回复”，但 topic 标题仍是“默认使用中文回复”，应把 topic 改为中性的“回复语言偏好”。

merge_topics：合并重复主题
- 多个 active topic 表达同一类记忆，继续分开会导致召回重复或维护成本上升。
- 只负责把 source topic 下的 active memories 迁移到 target topic，并把 source topic 标记为 merged。
- 不能修改任何 memory 的 summary/content/confidence；不能自动合并 memory；不能自动改 target topic 的 title/summary。
- 如果合并后 target topic 标题或概要不准确，必须另行输出 update_topic。
- 合并前应先 read_topic_detail 或 list_topic_memories 理解 target/source 内容；不要只凭标题相似就合并。

record_conflict_decision：记录冲突判断过程
- 当同一用户、同一作用域、同一语义槽位出现前后相反说法时使用。
- 如果根据用户最新明确纠正决定覆盖旧偏好，应记录 old_claim、new_claim、decision、reason。
- 它记录“为什么这样处理冲突”，不直接修改 memory 内容；通常与 update_memory 或 update_topic 搭配使用。
- 它可以记录 active 或 conflict memory 的冲突决策，但不能修改 memory 正文、状态或 topic。

组合动作规则：
- 用户偏好/工程规则被明确纠正时，通常需要三件套：
  1. update_memory：更新当前有效结论。
  2. update_topic：如果 topic 标题/概要仍是旧语义，改成中性或当前有效语义。
  3. record_conflict_decision：记录旧说法、新说法、采用新说法的原因。
- 如果只是补充同方向证据，用 merge_memory，不要记录冲突。
- 如果只是 topic 展示语义落后但 memory 内容已经正确，只用 update_topic + 必要的 record_conflict_decision。
- 如果冲突来自不同项目/芯片/工具链/版本条件，优先拆条件或拆记忆，不要直接覆盖。
- 执行 update_memory / merge_memory / merge_existing_memories / retire_memory / move_memory_topic / mark_conflict 后，必须自检 topic 标题/概要是否仍准确；如果只是补证据、合并重复 memory、移动归类或退役单条 memory，不要自动 update_topic；如果主题语义扩大、缩小或漂移，先读 topic 详情，再显式输出 update_topic。
- 执行 merge_topics 后，必须自检 target topic 标题/概要是否能覆盖合并后的范围；如果不能覆盖，另行 update_topic；merge_topics 本身不能改标题/概要。
- 不要让一个动作承担多个职责：新建只新建，更新只更新，主题只由 update_topic 修改，冲突原因只由 record_conflict_decision 记录。

状态权限矩阵：
- active memory：允许 update_memory、merge_memory、merge_existing_memories、retire_memory、move_memory_topic、mark_conflict、record_conflict_decision。
- conflict memory：只允许 record_conflict_decision；不能继续 update_memory、merge_memory 或 mark_conflict。
- merged / superseded / inactive memory：默认只读；本阶段不能写入。
- retired / merged / superseded / inactive memory：默认只读；本阶段不能写入。
- topic 只能通过 update_topic 修改 title/summary；create_memory、update_memory、merge_memory、merge_existing_memories、retire_memory、move_memory_topic、merge_topics、mark_conflict、record_conflict_decision、record_memory_relation 都不能隐式修改 topic。

noop：不沉淀
- 只是当前项目一次性参数或需求选择。
- 已有记忆已覆盖，且本轮没有新增证据或更好表述。
- 证据只是猜测或没有长期价值。

request_more_evidence：证据不足
- 当前 turn 摘要不足，必须回查 pointer。
- 或需要用户/后续验证才能确认。

拆分原则：
- 同症状不同根因，不要强行合并，应形成不同 memory。
- 不同项目、芯片、工具链、版本造成的差异，优先拆 scope/条件，不直接冲突。

====================
四、证据使用标准
====================

证据优先级：
1. 用户明确确认 > 助手总结。
2. 编译/测试/运行验证 > 推测。
3. 工具结果摘要 > 普通自然语言判断。
4. pointer 可补充原始证据，但不要把完整原文塞进 memory。

验证状态写法：
- “用户反馈现象”只能证明现象存在。
- “助手定位/推理”只能证明候选根因，除非有工具结果或用户确认支撑。
- “用户确认修复后正常 / 编译通过 / 烧录后验证正常 / 工具结果 verified=true”才算可写为已验证。
- 对运行现象来说，编译通过只能证明修复代码语法/链接无误，不能证明板上现象已修好。
- 未闭环的编译/运行信息只保留在 turns/evidence_items；本轮输出 noop，并用 chain_state 记录缺少什么证据，不能创建或更新正式技术 memory。
- 编译失败经验必须达到 compile_verified；运行失败经验必须有用户重新烧录/运行后的明确确认，达到 user_verified。
- 不要因为描述详细就自动提高 confidence，也不要把旧验证状态投射到新现象上。
- 编译失败类 memory 的完整目标结构是“症状 → 原因 → 有效修复 → 验证方式”；不要沉淀完整编译日志、完整 diff、临时尝试或一次性路径。

如果 core/candidate 摘要不足以判断是否达到上述准入门槛，或无法确定验证状态，应请求 expand_pointer。
如果已有 memory_index 数量有限或摘要看不出相关性，应先请求 search_related_memories，用关键词从主题标题、主题摘要、memory 摘要、memory 正文中召回候选。
如果可能要修改已有 memory，但目标 memory_id 不在 modifiable_memory_ids，必须先请求 read_memory_detail 或 read_topic_detail。
如果你认为“理想动作”是 merge_memory / update_memory / mark_conflict / record_conflict_decision，但当前还没有修改权限，不能因此输出 noop；必须先请求只读工具补足权限和详情。
如果主题下 memory 较多或不确定是否重复，应请求 list_topic_memories 先看轻量摘要，再决定是否 read_memory_detail。
如果需要判断新证据是补充、修正还是已处理过，应请求 read_memory_history 查看版本时间线。
如果要处理用户偏好变化或冲突，应请求 list_memory_conflicts 查看是否已有相同冲突记录。
如果怀疑某条 memory 已被合并、替代或与其他 memory 冲突，应请求 trace_memory_relations 查看关系链。
如果经过多轮工具调用后不确定当前 turn 的证据、evidence_id 或本轮是否足够支撑写入，可以请求 get_current_turn_evidence_summary 回看当前 turn 证据池。
如果当前 turn 涉及编译或上板故障链，最终判断前必须先请求 list_open_memory_chains；若现象/根因/修复确实命中同一既有链，原样带回其 chain_id，并读取该链全部关联历史 turn 的证据，不得丢弃旧环节。开放链若描述的是另一个症状、根因或修复对象，即使来自同一会话、同一模块或同一条多问题 turn，也不得复用该 chain_id。若没有明确命中或需要发现尚未标记的历史环节，再请求 list_previous_session_turns。选中相关 turn 后，再请求 read_session_turn_evidence，不要一次读取整个会话。确认为全新链时不要填写 chain_id。
chain_state 使用严格枚举，禁止发明近义状态：
- 非技术 turn：chain_type=none，state=not_applicable，missing_evidence=[]；也可以完全省略 chain_state。noop 是记忆动作，不是 chain state。
- 编译链：chain_type=compilation，state 只能是 failure_observed、cause_identified、fix_applied、fix_rejected、compile_verified。
- 运行链：chain_type=runtime，state 只能是 symptom_observed、cause_identified、fix_applied、waiting_user_validation、user_verified、fix_rejected。
- needs_verification、needs_user_verification、unverified、compile_unverified、diagnosis_only、no_chain、noop 都不是合法 state；必须根据已有证据选上面的精确枚举。
- 所有 chain_state 都必须包含 related_turn_ids 和 missing_evidence。未闭环技术链的 missing_evidence 必须非空；compile_verified、user_verified 和 none/not_applicable 的 missing_evidence 必须为空数组。
- user_verified 只允许用于用户明确说明重新烧录/运行后正常；助手声称已修复、代码已修改或编译通过均不得使用 user_verified。
- compile_verified 只允许用于失败修复后、对原始失败步骤的同类编译/链接/构建动作重跑成功；上游步骤成功不能证明下游失败消失。例如，原错误发生在烧录/下载时，仅有编译 0 Error 或 HEX 生成必须保持 fix_applied，不得判为 compile_verified。普通低价值 turn、路径问答和一次性任务不得因为助手文本提到编译而建立 compilation 链。
- 用户或编译结果明确否定上一轮修复时，必须复用同一 chain_id 并进入 fix_rejected；不得创建新链，也不得把被否定的修复写入正式技术 memory。
- compilation/fix_rejected 只能从 fix_applied 进入；runtime/fix_rejected 只能从 fix_applied 或 waiting_user_validation 进入。进入 fix_rejected 后，下一轮应重新定位原因或应用新的修复，直到获得对应的 compile_verified/user_verified。
- 续接被否定的修复时必须保留已有 related_turn_ids，并加入当前 turn；不能丢弃失败尝试的证据。
单链 turn 使用 chain_state；只有当同一 turn 真正包含多个独立故障动作时，才使用 chain_operations，两者不能同时出现。chain_operations 每项必须包含：
- operation_id：本次决策内唯一的 `op_` 小写标识，仅用于区分操作。
- mode：`update_existing` 或 `create_new`。更新旧链必须带 list_open_memory_chains 返回的 chain_id；创建新链禁止带 chain_id。
- chain_state：该独立故障链的完整状态；不允许 none/not_applicable。
- linked_change_indexes：该链闭环后所生成技术 memory 在 changes.changes 中的下标。未闭环新链必须是 []；每个技术 memory 必须且只能绑定一个链操作。
典型混合 turn：用户确认旧问题 A 已正常，同时报告不同的新现象 B。应输出两个操作：`update_existing + user_verified + linked_change_indexes:[旧经验下标]`，以及 `create_new + symptom_observed + linked_change_indexes:[]`。禁止用 B 的新现象重新打开 A，也禁止把 B 写进 A 的已验证 memory。
禁止凭 memory_index 摘要直接 update_memory / merge_memory / mark_conflict / record_conflict_decision。
禁止凭 search_related_memories 结果直接 update_memory / merge_memory / mark_conflict / record_conflict_decision；搜索结果只用于决定下一步读哪个 memory/topic 详情。
禁止凭 get_current_turn_evidence_summary 直接修改已有 memory；它只用于确认当前 turn 证据，修改已有 memory 仍必须先 read_memory_detail 或 read_topic_detail。
禁止凭 trace_memory_relations 直接修改已有 memory；关系链只用于理解来源/去向，修改已有 memory 仍必须先 read_memory_detail 或 read_topic_detail。

====================
五、输出格式
====================

如果需要补证据，输出：
{
  "schema_version": "memory-react-v1",
  "decision": "tool_requests",
  "reason": "中文说明为什么需要查",
  "tool_requests": [
    {"tool": "search_related_memories", "query": "症状/工具链/主题关键词", "reason": "为什么要搜旧记忆"},
    {"tool": "read_topic_detail", "topic_id": "已有 topic_id", "reason": "为什么查"},
    {"tool": "list_topic_memories", "topic_id": "已有 topic_id", "reason": "为什么列出主题下记忆"},
    {"tool": "read_memory_detail", "memory_id": "已有 memory_id", "reason": "为什么查"},
    {"tool": "read_memory_history", "memory_id": "已有 memory_id", "reason": "为什么查版本历史"},
    {"tool": "list_memory_conflicts", "memory_id": "已有 memory_id", "reason": "为什么查冲突记录"},
    {"tool": "expand_pointer", "raw_pointer_id": "本轮 pointer id", "reason": "为什么查"},
    {"tool": "get_current_turn_evidence_summary", "reason": "为什么需要回看当前 turn 证据"},
    {"tool": "list_open_memory_chains", "reason": "为什么需要查看同会话未闭环技术链", "limit": 10},
    {"tool": "list_previous_session_turns", "reason": "为什么需要查看同会话历史摘要", "limit": 10},
    {"tool": "read_session_turn_evidence", "turn_id": "必须来自历史摘要结果", "reason": "为什么该历史 turn 与当前问题相关"},
    {"tool": "trace_memory_relations", "memory_id": "已有 memory_id", "reason": "为什么需要查版本/关系链"}
  ],
  "changes": []
}

如果证据足够，输出：
{
  "schema_version": "memory-react-v1",
  "decision": "final_changes",
  "reason": "中文说明最终判断",
  "chain_state": {"chain_type":"runtime","state":"user_verified","related_turn_ids":["turn_xxx"],"missing_evidence":[]},
  "tool_requests": [],
  "changes": {
    "schema_version": "memory-changes-v1",
    "turn_id": "当前 turn_id",
    "source_turn_ids": ["当前 turn_id", "已经 read_session_turn_evidence 读取过的历史 turn_id"],
    "reason": "中文整体判断",
    "changes": []
  }
}

chain_state 合法示例：
- 非技术 noop：省略 chain_state，或 {"chain_type":"none","state":"not_applicable","related_turn_ids":["当前 turn_id"],"missing_evidence":[]}
- 编译未闭环：{"chain_type":"compilation","state":"fix_applied","related_turn_ids":["当前 turn_id"],"missing_evidence":["修复后的编译通过证据"]}
- 编译修复被否定：{"chain_type":"compilation","state":"fix_rejected","related_turn_ids":["历史 turn_id","当前 turn_id"],"missing_evidence":["重新定位根因、再次修复和编译通过证据"]}
- 运行待复测：{"chain_type":"runtime","state":"waiting_user_validation","related_turn_ids":["当前 turn_id"],"missing_evidence":["用户重新烧录运行后的确认"]}
- 运行修复被否定：{"chain_type":"runtime","state":"fix_rejected","related_turn_ids":["历史 turn_id","当前 turn_id"],"missing_evidence":["新的根因定位、有效修复和用户复测确认"]}

chain_operations 混合 turn 示例：
[
  {"operation_id":"op_close_mapping","mode":"update_existing","chain_state":{"chain_id":"chain_from_tool","chain_type":"runtime","state":"user_verified","related_turn_ids":["历史 turn_id","当前 turn_id"],"missing_evidence":[]},"linked_change_indexes":[0]},
  {"operation_id":"op_open_position","mode":"create_new","chain_state":{"chain_type":"runtime","state":"symptom_observed","related_turn_ids":["当前 turn_id"],"missing_evidence":["新现象的根因、修复和用户复测"]},"linked_change_indexes":[]}
]

memory-changes-v1 changes 可用 action：
- create_memory
- merge_memory
- merge_existing_memories
- update_memory
- retire_memory
- move_memory_topic
- merge_topics
- mark_conflict
- update_topic
- record_conflict_decision
- record_memory_relation
- noop
- request_more_evidence

create_memory 示例：
{
  "action": "create_memory",
  "reason": "为什么新增",
  "topic": {
    "topic_key": "embedded_stub_input_failure",
    "title": "嵌入式硬件输入桩导致假数据",
    "summary": "硬件输入函数返回固定值会让主流程跑通但输入恒定，导致下游状态机无响应。",
    "topic_type": "runtime_failure_experience",
    "scope": {"level": "domain", "domain": "embedded"}
  },
  "memory": {
    "stable_key": "embedded.stub_input.constant_value_downstream_stuck",
    "memory_type": "runtime_failure_experience",
    "summary": "硬件输入桩函数固定返回常量会导致下游控制逻辑一直处理假数据。",
    "content": "症状：编译烧录通过，但外设不随真实输入变化。触发条件：采样/驱动函数未读取真实硬件，而是返回固定常量。根因：主流程虽然执行，但输入数据恒定。解决：替换桩实现，按数据手册或现有驱动读取真实寄存器/外设。验证：改变真实输入后，下游状态、占空比或报警应随之变化。",
    "scope": {"level": "domain", "domain": "embedded"},
    "confidence": 0.9
  },
  "technical_closure": {
    "chain_type": "runtime",
    "state": "user_verified",
    "verification_evidence_ids": ["ev_user_confirmation"]
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "证明症状、根因或验证"}]
}

update_memory 示例：
{
  "action": "update_memory",
  "reason": "已有记忆方向正确，本轮提供了更通用表述",
  "target_memory_id": "mem_xxx",
  "memory": {
    "summary": "更新后的中文摘要",
    "content": "更新后的完整内容",
    "confidence": 0.92
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "补充证据"}]
}

merge_memory 示例：
{
  "action": "merge_memory",
  "reason": "同一模式，本轮只补充新证据或验证，不改变原结论",
  "target_memory_id": "mem_xxx",
  "memory": {
    "summary": "合并后的中文摘要，保持原核心结论，必要时加入新增适用条件",
    "content": "保留原有症状/根因/解决方法的简洁版本。新增证据：本轮补充了某项目中同类现象，或补充了用户确认/工具验证结果。不要粘贴完整日志、执行流或大段对话。",
    "confidence": 0.92
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "新增证据或验证"}]
}

merge_existing_memories 示例：
{
  "action": "merge_existing_memories",
  "reason": "多条已有 memory 语义重复，合并后减少召回噪声。",
  "target_memory_id": "mem_target",
  "source_memory_ids": ["mem_source_a", "mem_source_b"],
  "memory": {
    "summary": "合并后的中文摘要",
    "content": "合并后的完整内容，必须保留 target 和 sources 的有效结论，并去掉重复表述。",
    "confidence": 0.9
  },
  "conflict_disposition": {
    "conflict_ids": ["conflict_xxx"],
    "status": "resolved_by_merge",
    "reason": "source 被合并后，相关冲突由合并后的 target 结论解决。"
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "证明这些 memory 重复或同方向"}]
}

注意：merge_memory / merge_existing_memories / update_memory / mark_conflict 都必须包含 memory 对象，不能只给 target_memory_id 和 evidence_refs。

retire_memory 示例：
{
  "action": "retire_memory",
  "reason": "这条旧记忆已经不再作为当前有效结论使用，需要退役但保留历史。",
  "target_memory_id": "mem_xxx",
  "conflict_disposition": {
    "conflict_ids": ["conflict_xxx"],
    "status": "discarded_as_stale",
    "reason": "目标 memory 退役后，该冲突只保留历史意义。"
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "证明旧记忆已失效或不应继续作为当前有效记忆"}]
}

注意：retire_memory 不能包含 memory 对象，不能修改 topic；如果目标 memory 没有 active conflict，可以省略 conflict_disposition。

move_memory_topic 示例：
{
  "action": "move_memory_topic",
  "reason": "memory 内容正确，但原主题归类错误，移动到更准确的主题。",
  "target_memory_id": "mem_xxx",
  "from_topic_id": "topic_old",
  "to_topic_id": "topic_new",
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "证明该 memory 更适合目标主题"}]
}

注意：move_memory_topic 不能包含 memory 或 topic 对象，不能修改 memory 内容，也不能修改 topic 标题/概要。

merge_topics 示例：
{
  "action": "merge_topics",
  "reason": "两个主题语义重复，继续分开会造成召回重复，合并到更准确的目标主题。",
  "target_topic_id": "topic_target",
  "source_topic_ids": ["topic_source_a", "topic_source_b"],
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "证明这些主题表达同一类记忆"}]
}

注意：merge_topics 不能包含 memory 或 topic 对象，不能修改 memory 内容，不能自动改 target topic 标题/概要；如需改摘要，另行 update_topic。

update_topic 示例：
{
  "action": "update_topic",
  "reason": "memory 已从旧偏好更新为新偏好，原主题标题不再准确",
  "target_topic_id": "topic_xxx",
  "topic": {
    "title": "回复语言偏好",
    "summary": "记录用户当前偏好的回复语言，以及前后纠正造成的当前有效偏好。"
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "用户明确提出新的回复语言偏好"}]
}

record_conflict_decision 示例：
{
  "action": "record_conflict_decision",
  "reason": "同一用户回复语言偏好前后相反，需要记录为何采用新偏好",
  "target_memory_id": "mem_xxx",
  "conflict": {
    "conflict_type": "user_preference_changed",
    "old_claim": "用户希望默认中文回复",
    "new_claim": "用户希望默认英文回复",
    "decision": "采用更新的明确用户偏好，并保留冲突记录",
    "reason": "新证据是用户对未来回复语言的明确要求，且与旧偏好属于同一语义槽位。"
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "新偏好证据"}]
}

record_memory_relation 示例：
{
  "action": "record_memory_relation",
  "reason": "用户明确纠正旧记忆，新记忆替代旧记忆",
  "relation": {
    "from_memory_id": "mem_old",
    "to_memory_id": "mem_new",
    "relation_type": "supersedes",
    "reason": "同一语义槽位下，新证据来自更晚的用户明确纠正。"
  },
  "evidence_refs": [{"evidence_id": "ev_xxx", "reason": "用户纠正证据"}]
}

noop 示例：
{
  "action": "noop",
  "reason": "只是当前项目一次性参数，不能沉淀为长期记忆",
  "evidence_refs": []
}

====================
六、硬性要求
====================

- decision=tool_requests 时，tool_requests 必须非空，changes 必须是空数组。
- decision=final_changes 时，tool_requests 必须是空数组，changes 必须是完整 memory-changes-v1 对象。
- changes.source_turn_ids 可省略；单 turn 时默认只有当前 turn。跨 turn 时必须包含当前 turn，且历史 turn 只能来自 turn_access_state.observed_turn_ids。
- 单链使用 chain_state；多链混合 turn 使用 chain_operations；两者互斥。技术 memory 写入时必须由匹配的闭环 chain_state 支撑，多链时还必须通过 linked_change_indexes 精确绑定。
- compile_failure_experience 的 create/merge/update 必须带 technical_closure，state 只能是 compile_verified。
- runtime_failure_experience 的 create/merge/update 必须带 technical_closure，state 只能是 user_verified，验证证据必须包含用户确认。
- 每轮最多请求 3 个工具。
- 只能请求输入中真实存在的 topic_id / memory_id / raw_pointer_id。
- 不能发明 evidence_id、memory_id、topic_id、raw_pointer_id。
- stable_key/topic_key 不可修改，不要在 reason 中把“修正 key”作为 update_memory/update_topic 的理由；只能修改工具 schema 允许的字段。
- search_related_memories 的 query 必须来自当前 turn 证据或已有 memory/topic 摘要中的真实词语，不能发明不存在的问题。
- update_memory / merge_memory / merge_existing_memories / retire_memory / move_memory_topic / mark_conflict 的 target_memory_id 必须在 modifiable_memory_ids 中；否则先请求只读工具。
- record_conflict_decision 的 target_memory_id 也必须在 modifiable_memory_ids 中；否则先请求只读工具。
- 不能把“目标 memory 未读取、暂时没有修改权限”当成 noop 理由；应先请求 read_memory_detail / read_topic_detail / list_topic_memories。
- update_topic 的 target_topic_id 必须来自已知 topic_id；如果要判断主题是否应改名，优先先 read_topic_detail。
- merge_topics 的 target_topic_id/source_topic_ids 必须来自已知 active topic_id；source 不能包含 target；如需确认主题下记忆，先 list_topic_memories。
- 对用户明确纠正导致的同一偏好反转，优先输出 update_memory + update_topic + record_conflict_decision 的组合；如果 memory 已经是最新结论，则至少输出 update_topic + record_conflict_decision。
- 如果已有 memory 覆盖本轮信息，应 merge/update/noop，不要重复 create。
- 模型不能写数据库，不能输出 SQL，不能请求未列出的工具。
"""


REACT_REPAIR_PROMPT = """你是 Memory Lab ReAct JSON 修复器。
你会收到上一次模型输出、程序校验错误和允许 ID。
只修复 JSON schema、字段名、字段结构和引用 ID，不要改变原本语义判断。
如果错误是 unsupported fields，必须删除这些字段，不能换成别的未知字段。
特别注意：
- chain_state 是严格协议字段，允许值只有：none/not_applicable；compilation/failure_observed、cause_identified、fix_applied、fix_rejected、compile_verified；runtime/symptom_observed、cause_identified、fix_applied、waiting_user_validation、user_verified、fix_rejected。
- chain_state 与 chain_operations 互斥。同一 turn 同时更新旧链并创建新链时，必须改用 chain_operations；每项只允许 operation_id、mode、chain_state、linked_change_indexes。
- chain_operations.mode=update_existing 必须带工具返回的 chain_id；mode=create_new 必须省略 chain_id。linked_change_indexes 只能引用 changes.changes 中带 technical_closure 的技术 memory，且不能被多个操作重复引用。
- needs_verification、needs_user_verification、unverified、compile_unverified、diagnosis_only、no_chain、noop 都是非法 state。根据 previous_output 的原意改成最接近的合法状态；noop 是 action，不是 state。
- chain_state 必须含 chain_type、state、related_turn_ids、missing_evidence。未闭环技术链 missing_evidence 必须非空；compile_verified、user_verified、none/not_applicable 时必须为 []。
- 非技术 turn 应省略 chain_state，或使用 none/not_applicable；不要因为助手文本里出现“编译”就建立技术链。
- user_verified 必须有用户测试、重新烧录或运行后确认正常；用户明确说“我测试了，功能已实现/问题消失/现在正常”可以作为用户确认证据，不要求原话必须含“烧录”。不能由助手的修复声明或编译通过替代。compile_verified 必须是失败修复后、对原始失败步骤的同类编译/链接/构建动作重跑成功；如果原失败发生在烧录/下载，仅有编译 0 Error 或生成 HEX 仍是 fix_applied，不是 compile_verified。
- fix_rejected 必须有用户否定、重新编译仍失败或等价的失败证据；不能因为“尚未验证”而使用 fix_rejected，尚未验证应保留 fix_applied 或 waiting_user_validation。
- 如果校验错误要求先完成 list_open_memory_chains，不要继续输出技术 final_changes；改为请求 list_open_memory_chains 的 tool_requests。
- decision=tool_requests 时必须同时输出 `changes: []`，不能省略 changes。
- read_session_turn_evidence 只能读取 available_ids.discovered_turn_ids 中的历史 turn；current_turn_id 已在当前证据包内，禁止用历史工具读取当前 turn。
- available_ids.observed_turn_ids 中的历史 turn 已经读取完成，禁止重复请求 read_session_turn_evidence；必须直接使用 observations 中已有证据继续判断。
- read_session_turn_evidence.layers 如填写，只能是 `['core']`、`['core','candidate']` 或 `['core','candidate','pointer']` 这类由 core/candidate/pointer 组成的非空去重数组；不存在 raw/summary 层。不确定时省略 layers。
- noop 只允许 action、reason、evidence_refs，不能带 memory_id、target_memory_id、topic_id、memory。
- create_memory 只能二选一 topic 或 topic_id。
- create_memory 不能修改已有 topic；已有主题必须使用 topic_id，改主题只能使用 update_topic。
- merge_memory/merge_existing_memories/update_memory/retire_memory/move_memory_topic/mark_conflict 必须使用 target_memory_id，不能使用 memory_id。
- merge_memory/merge_existing_memories/update_memory/retire_memory/move_memory_topic/mark_conflict 不能修改 topic；如需修改主题标题/概要，另行输出 update_topic。
- merge_memory/merge_existing_memories/update_memory/mark_conflict 必须包含 memory 对象，memory 里必须包含 summary、content、confidence。
- create_memory.memory.memory_type 只能是 user_preference、compile_failure_experience、runtime_failure_experience；旧类型或近义类型必须改成这三类之一，不能发明新类型。
- compile_failure_experience 的 create/merge/update 必须带 technical_closure={"chain_type":"compilation","state":"compile_verified","verification_evidence_ids":[...]}。
- runtime_failure_experience 的 create/merge/update 必须带 technical_closure={"chain_type":"runtime","state":"user_verified","verification_evidence_ids":[...]}，且验证证据必须是用户确认。
- 写技术 memory 时，单链的顶层 chain_state 必须与 technical_closure 的 chain_type/state 一致；多链时必须由唯一 chain_operation 绑定且状态一致。
- merge_existing_memories 必须包含 source_memory_ids；source 不能包含 target。
- retire_memory 不能包含 memory 对象；如目标 memory 有 active conflict，必须包含 conflict_disposition。
- move_memory_topic 不能包含 memory 或 topic 对象，必须包含 from_topic_id 和 to_topic_id。
- merge_topics 不能包含 memory 或 topic 对象，必须包含 target_topic_id 和 source_topic_ids。
- update_topic 必须使用 target_topic_id 和 topic 对象，topic 只能包含 title、summary。
- record_conflict_decision 必须使用 target_memory_id 和 conflict 对象。
- 如果 merge_memory 缺少 memory 对象，必须基于已读取的目标 memory 正文和新增证据补出合并后的完整 memory 对象；不要删除该 action。
- tool_requests 和 final_changes 不能混用。
tool_requests 的最小合法完整示例：
{"schema_version":"memory-react-v1","decision":"tool_requests","reason":"补充历史证据","tool_requests":[{"tool":"read_session_turn_evidence","turn_id":"必须来自 discovered_turn_ids 且不能是 current_turn_id","reason":"读取相关历史 turn","layers":["core","candidate","pointer"]}],"changes":[]}
只能输出完整 JSON 对象，不要输出 Markdown、解释、代码块或额外文本。
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_run_log(
    *,
    run_log: Mapping[str, Any],
    log_dir: Path | None = DEFAULT_REACT_RUN_DIR,
) -> dict[str, str]:
    log_dir = log_dir or DEFAULT_REACT_RUN_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(run_log["run_id"])
    path = log_dir / f"{run_id}.json"
    path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = log_dir / "latest.json"
    try:
        shutil.copyfile(path, latest)
    except OSError:
        latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"run_log_path": str(path), "latest_log_path": str(latest)}


def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _collect_available_ids(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    layers = (payload.get("turn_evidence") or {}).get("layers") or {}
    pointer_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for layer_name in ("core", "candidate", "pointer"):
        for item in layers.get(layer_name) or []:
            if isinstance(item, Mapping):
                evidence_id = item.get("evidence_id")
                if evidence_id:
                    evidence_ids.add(str(evidence_id))
                raw_pointer_id = item.get("raw_pointer_id")
                if raw_pointer_id:
                    pointer_ids.add(str(raw_pointer_id))
    for item in layers.get("pointer") or []:
        if isinstance(item, Mapping) and item.get("evidence_id"):
            pointer_ids.add(str(item["evidence_id"]))

    topic_ids = {
        str(item["topic_id"])
        for item in (payload.get("topic_index") or {}).get("topics") or []
        if isinstance(item, Mapping) and item.get("topic_id")
    }
    memory_index = payload.get("memory_index") or {}
    memory_ids = {
        str(item["memory_id"])
        for item in memory_index.get("memories") or []
        if isinstance(item, Mapping) and item.get("memory_id")
    }
    for item in memory_index.get("topics") or []:
        if isinstance(item, Mapping) and item.get("topic_id"):
            topic_ids.add(str(item["topic_id"]))
    return {
        "evidence_ids": evidence_ids,
        "pointer_ids": pointer_ids,
        "topic_ids": topic_ids,
        "memory_ids": memory_ids,
        "discovered_turn_ids": set(),
        "observed_turn_ids": set(),
        "historical_evidence_ids": set(),
        "open_chain_scan_completed": set(),
        "open_chain_ids": set(),
    }


def _ids_for_log(ids: Mapping[str, set[str]]) -> dict[str, list[str]]:
    return {key: sorted(value) for key, value in ids.items()}


def _tool_key(request: Mapping[str, Any]) -> tuple[str, str]:
    tool = str(request.get("tool") or "")
    if tool == "read_topic_detail":
        return tool, str(request.get("topic_id") or "")
    if tool == "list_topic_memories":
        return tool, str(request.get("topic_id") or "")
    if tool == "read_memory_detail":
        return tool, str(request.get("memory_id") or "")
    if tool == "read_memory_history":
        return tool, str(request.get("memory_id") or "")
    if tool == "list_memory_conflicts":
        return tool, str(request.get("memory_id") or request.get("topic_id") or "")
    if tool == "expand_pointer":
        return tool, str(request.get("raw_pointer_id") or "")
    if tool == "search_related_memories":
        parts = [
            str(request.get("query") or ""),
            str(request.get("memory_type") or ""),
            str(request.get("domain") or ""),
            str(request.get("project_key") or ""),
        ]
        return tool, "|".join(parts)
    if tool == "get_current_turn_evidence_summary":
        return tool, "current_turn"
    if tool == "list_open_memory_chains":
        return tool, "current_session"
    if tool == "list_previous_session_turns":
        return tool, "current_session"
    if tool == "read_session_turn_evidence":
        return tool, str(request.get("turn_id") or "")
    if tool == "trace_memory_relations":
        return tool, str(request.get("memory_id") or "")
    return tool, ""


def validate_react_decision(
    decision: Mapping[str, Any],
    *,
    turn_id: str,
    available_ids: Mapping[str, set[str]],
    observed_memory_ids: set[str] | None = None,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    max_tool_requests: int = 3,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {"schema_version", "decision", "reason", "tool_requests", "changes"}
    allowed = required | {"chain_state", "chain_operations"}
    extra = set(decision) - allowed
    missing = required - set(decision)
    if extra:
        errors.append(f"unexpected top-level fields: {sorted(extra)}")
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
    if decision.get("schema_version") != MEMORY_REACT_SCHEMA_VERSION:
        errors.append("schema_version must be memory-react-v1")
    if decision.get("decision") not in {"tool_requests", "final_changes"}:
        errors.append("decision must be tool_requests or final_changes")
    if not _non_empty_string(decision.get("reason")):
        errors.append("reason must be a non-empty string")

    chain_state = decision.get("chain_state")
    raw_chain_operations = decision.get("chain_operations")
    chain_operations: list[dict[str, Any]] = []
    if chain_state is not None and raw_chain_operations is not None:
        errors.append("chain_state and chain_operations are mutually exclusive")
    if chain_state is not None:
        try:
            chain_state = validate_chain_state(
                chain_state,
                current_turn_id=turn_id,
                observed_turn_ids=available_ids.get("observed_turn_ids") or set(),
            )
        except MemoryChainStateValidationError as error:
            errors.extend(error.errors)
    if raw_chain_operations is not None:
        try:
            chain_operations = validate_chain_operations(
                raw_chain_operations,
                current_turn_id=turn_id,
                observed_turn_ids=available_ids.get("observed_turn_ids") or set(),
            )
        except MemoryChainStateValidationError as error:
            errors.extend(error.errors)

    technical_chain_states = [
        state
        for state in (
            [chain_state] if isinstance(chain_state, Mapping) else []
        ) + [
            operation.get("chain_state")
            for operation in chain_operations
            if isinstance(operation.get("chain_state"), Mapping)
        ]
        if isinstance(state, Mapping)
        and state.get("chain_type") in {"compilation", "runtime"}
    ]
    if decision.get("decision") == "final_changes" and technical_chain_states:
        if "current_session" not in (
            available_ids.get("open_chain_scan_completed") or set()
        ):
            errors.append(
                "technical final decision requires a completed list_open_memory_chains observation"
            )
        for technical_index, technical_state in enumerate(technical_chain_states):
            chain_id = str(technical_state.get("chain_id") or "")
            if not chain_id:
                continue
            prefix = (
                "chain_state"
                if not chain_operations
                else f"chain_operations[{technical_index}].chain_state"
            )
            if chain_id not in (available_ids.get("open_chain_ids") or set()):
                errors.append(
                    f"{prefix}.chain_id must come from a completed list_open_memory_chains observation"
                )
            required_turn_ids = available_ids.get(
                f"open_chain_required::{chain_id}"
            ) or set()
            related_turn_ids = {
                str(item) for item in technical_state.get("related_turn_ids") or []
            }
            missing_turn_ids = set(required_turn_ids) - related_turn_ids
            if missing_turn_ids:
                errors.append(
                    f"{prefix}.related_turn_ids cannot drop turns from the selected open chain: "
                    + ", ".join(sorted(missing_turn_ids))
                )

    tool_requests = decision.get("tool_requests")
    if not isinstance(tool_requests, list):
        errors.append("tool_requests must be an array")
        tool_requests = []
    elif len(tool_requests) > max_tool_requests:
        errors.append(f"tool_requests exceeds max {max_tool_requests}")

    changes = decision.get("changes")
    mode = decision.get("decision")
    if mode == "tool_requests":
        if not tool_requests:
            errors.append("tool_requests decision requires at least one tool request")
        if changes != []:
            errors.append("tool_requests decision requires changes=[]")
    elif mode == "final_changes":
        if tool_requests:
            errors.append("final_changes decision requires tool_requests=[]")
        if not _is_object(changes):
            errors.append("final_changes decision requires changes object")

    seen: set[tuple[str, str]] = set()
    pointer_ids = available_ids.get("pointer_ids") or set()
    topic_ids = available_ids.get("topic_ids") or set()
    memory_ids = available_ids.get("memory_ids") or set()
    for index, request in enumerate(tool_requests):
        prefix = f"tool_requests[{index}]"
        if not _is_object(request):
            errors.append(f"{prefix} must be an object")
            continue
        tool = request.get("tool")
        if tool not in ALLOWED_TOOLS:
            errors.append(f"{prefix}.tool is not allowed")
            continue
        allowed_by_tool = {
            "read_topic_detail": {"tool", "topic_id", "reason"},
            "list_topic_memories": {"tool", "topic_id", "reason", "limit"},
            "read_memory_detail": {"tool", "memory_id", "reason"},
            "read_memory_history": {"tool", "memory_id", "reason", "limit"},
            "list_memory_conflicts": {
                "tool",
                "memory_id",
                "topic_id",
                "reason",
                "limit",
                "status",
                "include_resolved",
            },
            "expand_pointer": {"tool", "raw_pointer_id", "reason"},
            "search_related_memories": {
                "tool",
                "query",
                "reason",
                "memory_type",
                "domain",
                "project_key",
                "limit",
            },
            "get_current_turn_evidence_summary": {
                "tool",
                "reason",
                "max_items_per_layer",
                "max_chars",
            },
            "list_open_memory_chains": {"tool", "reason", "limit"},
            "list_previous_session_turns": {
                "tool",
                "reason",
                "limit",
                "preview_chars",
            },
            "read_session_turn_evidence": {
                "tool",
                "turn_id",
                "reason",
                "layers",
                "max_chars",
            },
            "trace_memory_relations": {
                "tool",
                "memory_id",
                "reason",
                "direction",
                "max_depth",
            },
        }[str(tool)]
        required_by_tool = {
            "read_topic_detail": {"tool", "topic_id", "reason"},
            "list_topic_memories": {"tool", "topic_id", "reason"},
            "read_memory_detail": {"tool", "memory_id", "reason"},
            "read_memory_history": {"tool", "memory_id", "reason"},
            "list_memory_conflicts": {"tool", "reason"},
            "expand_pointer": {"tool", "raw_pointer_id", "reason"},
            "search_related_memories": {"tool", "query", "reason"},
            "get_current_turn_evidence_summary": {"tool", "reason"},
            "list_open_memory_chains": {"tool", "reason"},
            "list_previous_session_turns": {"tool", "reason"},
            "read_session_turn_evidence": {"tool", "turn_id", "reason"},
            "trace_memory_relations": {"tool", "memory_id", "reason"},
        }[str(tool)]
        extra_request = set(request) - allowed_by_tool
        missing_request = required_by_tool - set(request)
        if extra_request:
            errors.append(f"{prefix} has unsupported fields: {sorted(extra_request)}")
        if missing_request:
            errors.append(f"{prefix} missing fields: {sorted(missing_request)}")
        if not _non_empty_string(request.get("reason")):
            errors.append(f"{prefix}.reason must be a non-empty string")
        if tool == "search_related_memories":
            if not _non_empty_string(request.get("query")):
                errors.append(f"{prefix}.query must be a non-empty string")
            if "memory_type" in request and request.get("memory_type") not in ALLOWED_MEMORY_TYPES:
                errors.append(
                    f"{prefix}.memory_type must be one of {', '.join(sorted(ALLOWED_MEMORY_TYPES))}"
                )
            if "limit" in request:
                try:
                    limit_value = int(request.get("limit"))
                    if limit_value < 1 or limit_value > 20:
                        errors.append(f"{prefix}.limit must be between 1 and 20")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.limit must be an integer")
        if tool == "get_current_turn_evidence_summary":
            if "max_items_per_layer" in request:
                try:
                    max_items_value = int(request.get("max_items_per_layer"))
                    if max_items_value < 1 or max_items_value > 50:
                        errors.append(f"{prefix}.max_items_per_layer must be between 1 and 50")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.max_items_per_layer must be an integer")
            if "max_chars" in request:
                try:
                    max_chars_value = int(request.get("max_chars"))
                    if max_chars_value < 80 or max_chars_value > 4000:
                        errors.append(f"{prefix}.max_chars must be between 80 and 4000")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.max_chars must be an integer")
        if tool == "list_open_memory_chains":
            if "limit" in request:
                try:
                    open_chain_limit = int(request.get("limit"))
                    if open_chain_limit < 1 or open_chain_limit > 20:
                        errors.append(f"{prefix}.limit must be between 1 and 20")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.limit must be an integer")
        if tool == "list_previous_session_turns":
            if "limit" in request:
                try:
                    history_limit = int(request.get("limit"))
                    if history_limit < 1 or history_limit > 20:
                        errors.append(f"{prefix}.limit must be between 1 and 20")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.limit must be an integer")
            if "preview_chars" in request:
                try:
                    preview_chars = int(request.get("preview_chars"))
                    if preview_chars < 80 or preview_chars > 800:
                        errors.append(f"{prefix}.preview_chars must be between 80 and 800")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.preview_chars must be an integer")
        if tool == "read_session_turn_evidence":
            target_turn_id = str(request.get("turn_id") or "")
            if target_turn_id not in (available_ids.get("discovered_turn_ids") or set()):
                errors.append(
                    f"{prefix}.turn_id must come from a completed list_open_memory_chains or list_previous_session_turns observation"
                )
            if target_turn_id in (available_ids.get("observed_turn_ids") or set()):
                errors.append(
                    f"{prefix}.turn_id was already read; use the existing observation instead of repeating the tool request"
                )
            if "layers" in request:
                layers = request.get("layers")
                if (
                    not isinstance(layers, list)
                    or not layers
                    or any(layer not in {"core", "candidate", "pointer"} for layer in layers)
                    or len(set(layers)) != len(layers)
                ):
                    errors.append(
                        f"{prefix}.layers must be a non-empty unique array of core/candidate/pointer"
                    )
            if "max_chars" in request:
                try:
                    history_max_chars = int(request.get("max_chars"))
                    if history_max_chars < 80 or history_max_chars > 8000:
                        errors.append(f"{prefix}.max_chars must be between 80 and 8000")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.max_chars must be an integer")
        if tool == "trace_memory_relations":
            if request.get("memory_id") not in memory_ids:
                errors.append(f"{prefix}.memory_id does not exist or is not active")
            if "direction" in request and request.get("direction") not in {"backward", "forward", "both"}:
                errors.append(f"{prefix}.direction must be backward, forward or both")
            if "max_depth" in request:
                try:
                    max_depth_value = int(request.get("max_depth"))
                    if max_depth_value < 1 or max_depth_value > 5:
                        errors.append(f"{prefix}.max_depth must be between 1 and 5")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.max_depth must be an integer")
        if tool == "read_topic_detail" and request.get("topic_id") not in topic_ids:
            errors.append(f"{prefix}.topic_id does not exist or is not active")
        if tool == "list_topic_memories":
            if request.get("topic_id") not in topic_ids:
                errors.append(f"{prefix}.topic_id does not exist or is not active")
            if "limit" in request:
                try:
                    limit_value = int(request.get("limit"))
                    if limit_value < 1 or limit_value > 50:
                        errors.append(f"{prefix}.limit must be between 1 and 50")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.limit must be an integer")
        if tool == "read_memory_detail" and request.get("memory_id") not in memory_ids:
            errors.append(f"{prefix}.memory_id does not exist or is not active")
        if tool == "read_memory_history":
            if request.get("memory_id") not in memory_ids:
                errors.append(f"{prefix}.memory_id does not exist or is not active")
            if "limit" in request:
                try:
                    limit_value = int(request.get("limit"))
                    if limit_value < 1 or limit_value > 50:
                        errors.append(f"{prefix}.limit must be between 1 and 50")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.limit must be an integer")
        if tool == "list_memory_conflicts":
            has_memory = _non_empty_string(request.get("memory_id"))
            has_topic = _non_empty_string(request.get("topic_id"))
            if has_memory == has_topic:
                errors.append(f"{prefix} must contain exactly one of memory_id/topic_id")
            if has_memory and request.get("memory_id") not in memory_ids:
                errors.append(f"{prefix}.memory_id does not exist or is not active")
            if has_topic and request.get("topic_id") not in topic_ids:
                errors.append(f"{prefix}.topic_id does not exist or is not active")
            if "limit" in request:
                try:
                    limit_value = int(request.get("limit"))
                    if limit_value < 1 or limit_value > 50:
                        errors.append(f"{prefix}.limit must be between 1 and 50")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.limit must be an integer")
            if "status" in request and request.get("status") not in CONFLICT_RECORD_STATUSES:
                errors.append(
                    f"{prefix}.status must be one of {', '.join(sorted(CONFLICT_RECORD_STATUSES))}"
                )
            if "include_resolved" in request and not isinstance(request.get("include_resolved"), bool):
                errors.append(f"{prefix}.include_resolved must be a boolean")
        if tool == "expand_pointer" and request.get("raw_pointer_id") not in pointer_ids:
            errors.append(f"{prefix}.raw_pointer_id does not belong to this turn")
        key = _tool_key(request)
        if key in seen:
            errors.append(f"{prefix} duplicates tool request {key[0]}:{key[1]}")
        seen.add(key)

    if mode == "final_changes" and _is_object(changes):
        try:
            normalized_changes = validate_memory_changes(changes, pg_config_path=pg_config_path)
            if normalized_changes.get("turn_id") != turn_id:
                errors.append("changes.turn_id must equal current turn_id")
            source_turn_ids = normalized_changes.get("source_turn_ids")
            if source_turn_ids is None:
                source_turn_ids = [turn_id]
            allowed_source_turn_ids = {
                turn_id,
                *(available_ids.get("observed_turn_ids") or set()),
            }
            if isinstance(source_turn_ids, list):
                unauthorized_source_turn_ids = {
                    str(source_turn_id)
                    for source_turn_id in source_turn_ids
                    if str(source_turn_id) not in allowed_source_turn_ids
                }
                if unauthorized_source_turn_ids:
                    errors.append(
                        "changes.source_turn_ids contain historical turns that were not read by read_session_turn_evidence: "
                        + ", ".join(sorted(unauthorized_source_turn_ids))
                    )
            plan_changes = normalized_changes.get("changes") or []
            technical_change_indexes = [
                change_index
                for change_index, change in enumerate(plan_changes)
                if isinstance(change, Mapping)
                and isinstance(change.get("technical_closure"), Mapping)
            ]
            technical_closures = [
                plan_changes[change_index].get("technical_closure")
                for change_index in technical_change_indexes
            ]
            if chain_operations and not technical_closures:
                for operation_index, operation in enumerate(chain_operations):
                    if operation.get("linked_change_indexes"):
                        errors.append(
                            f"chain_operations[{operation_index}].linked_change_indexes must be empty when no technical memory is written"
                        )
            if technical_closures:
                if chain_operations:
                    linked_indexes: set[int] = set()
                    for operation_index, operation in enumerate(chain_operations):
                        operation_state = operation.get("chain_state") or {}
                        for linked_index in operation.get("linked_change_indexes") or []:
                            if linked_index >= len(plan_changes):
                                errors.append(
                                    f"chain_operations[{operation_index}].linked_change_indexes contains out-of-range index {linked_index}"
                                )
                                continue
                            linked_change = plan_changes[linked_index]
                            closure = (
                                linked_change.get("technical_closure")
                                if isinstance(linked_change, Mapping)
                                else None
                            )
                            if not isinstance(closure, Mapping):
                                errors.append(
                                    f"chain_operations[{operation_index}].linked_change_indexes[{linked_index}] must reference a technical memory change"
                                )
                                continue
                            linked_indexes.add(linked_index)
                            if closure.get("chain_type") != operation_state.get("chain_type"):
                                errors.append(
                                    f"chain_operations[{operation_index}] linked technical_closure.chain_type must match chain_state.chain_type"
                                )
                            if closure.get("state") != operation_state.get("state"):
                                errors.append(
                                    f"chain_operations[{operation_index}] linked technical_closure.state must match chain_state.state"
                                )
                    missing_links = set(technical_change_indexes) - linked_indexes
                    if missing_links:
                        errors.append(
                            "every technical memory change must be linked by exactly one chain operation: "
                            + ", ".join(str(index) for index in sorted(missing_links))
                        )
                elif not isinstance(chain_state, Mapping):
                    errors.append("chain_state or chain_operations is required when writing a technical memory")
                else:
                    for closure_index, closure in enumerate(technical_closures):
                        if closure.get("chain_type") != chain_state.get("chain_type"):
                            errors.append(
                                f"technical_closure[{closure_index}].chain_type must match chain_state.chain_type"
                            )
                        if closure.get("state") != chain_state.get("state"):
                            errors.append(
                                f"technical_closure[{closure_index}].state must match chain_state.state"
                            )
            observed = observed_memory_ids or set()
            for change_index, change in enumerate(normalized_changes.get("changes") or []):
                if not isinstance(change, Mapping):
                    continue
                if change.get("action") in {
                    "merge_memory",
                    "merge_existing_memories",
                    "update_memory",
                    "retire_memory",
                    "move_memory_topic",
                    "mark_conflict",
                    "record_conflict_decision",
                }:
                    target_memory_id = str(change.get("target_memory_id") or "")
                    if target_memory_id and target_memory_id not in observed:
                        errors.append(
                            f"changes.changes[{change_index}].target_memory_id must be read by read_memory_detail or read_topic_detail before modifying or recording conflict for existing memory"
                        )
                    if change.get("action") == "merge_existing_memories":
                        for source_memory_id in change.get("source_memory_ids") or []:
                            source_memory_id = str(source_memory_id)
                            if source_memory_id and source_memory_id not in observed:
                                errors.append(
                                    f"changes.changes[{change_index}].source_memory_ids must be read by read_memory_detail or read_topic_detail before merging existing memories"
                                )
        except MemoryChangesValidationError as error:
            errors.extend([f"changes.{item}" for item in error.errors])

    if errors:
        raise MemoryReactValidationError(errors)
    normalized_decision = dict(decision)
    if isinstance(chain_state, Mapping):
        normalized_decision["chain_state"] = dict(chain_state)
    if raw_chain_operations is not None and chain_operations:
        normalized_decision["chain_operations"] = chain_operations
    return normalized_decision


def _execute_tool_request(
    request: Mapping[str, Any],
    *,
    source_database: Path,
    pg_config_path: Path,
    turn_id: str,
    max_chars: int,
) -> dict[str, Any]:
    tool = str(request["tool"])
    if tool == "read_topic_detail":
        return read_topic_detail(
            pg_config_path=pg_config_path,
            topic_id=str(request["topic_id"]),
            max_chars=max_chars,
        )
    if tool == "list_topic_memories":
        return list_topic_memories(
            pg_config_path=pg_config_path,
            topic_id=str(request["topic_id"]),
            limit=int(request.get("limit") or 20),
        )
    if tool == "read_memory_detail":
        result = read_memory_detail(
            pg_config_path=pg_config_path,
            memory_id=str(request["memory_id"]),
        )
        memory = result.get("memory")
        if isinstance(memory, Mapping) and isinstance(memory.get("content"), str):
            content = memory["content"]
            result = dict(result)
            memory = dict(memory)
            memory["content"] = content[:max_chars]
            memory["content_truncated"] = len(content) > max_chars
            result["memory"] = memory
        return result
    if tool == "read_memory_history":
        return read_memory_history(
            pg_config_path=pg_config_path,
            memory_id=str(request["memory_id"]),
            limit=int(request.get("limit") or 10),
        )
    if tool == "list_memory_conflicts":
        return list_memory_conflicts(
            pg_config_path=pg_config_path,
            memory_id=str(request["memory_id"]) if request.get("memory_id") else None,
            topic_id=str(request["topic_id"]) if request.get("topic_id") else None,
            status_filter=str(request["status"]) if request.get("status") else None,
            include_resolved=bool(request.get("include_resolved") or False),
            limit=int(request.get("limit") or 20),
        )
    if tool == "expand_pointer":
        return expand_pointer(
            source_database,
            pg_config_path=pg_config_path,
            raw_pointer_id=str(request["raw_pointer_id"]),
            max_chars=max_chars,
        )
    if tool == "search_related_memories":
        return search_related_memories(
            pg_config_path=pg_config_path,
            query=str(request["query"]),
            memory_type=str(request["memory_type"]) if request.get("memory_type") else None,
            domain=str(request["domain"]) if request.get("domain") else None,
            project_key=str(request["project_key"]) if request.get("project_key") else None,
            limit=int(request.get("limit") or 8),
        )
    if tool == "get_current_turn_evidence_summary":
        return get_current_turn_evidence_summary(
            pg_config_path=pg_config_path,
            turn_id=turn_id,
            max_items_per_layer=int(request.get("max_items_per_layer") or 12),
            max_chars=int(request.get("max_chars") or max_chars),
        )
    if tool == "list_open_memory_chains":
        return list_open_memory_chains(
            pg_config_path=pg_config_path,
            current_turn_id=turn_id,
            limit=int(request.get("limit") or 10),
        )
    if tool == "list_previous_session_turns":
        return list_previous_session_turns(
            pg_config_path=pg_config_path,
            current_turn_id=turn_id,
            limit=int(request.get("limit") or 10),
            preview_chars=int(request.get("preview_chars") or 320),
        )
    if tool == "read_session_turn_evidence":
        return read_session_turn_evidence(
            pg_config_path=pg_config_path,
            current_turn_id=turn_id,
            target_turn_id=str(request["turn_id"]),
            layers=(
                [str(layer) for layer in request.get("layers") or []]
                if request.get("layers")
                else None
            ),
            max_chars=int(request.get("max_chars") or max_chars),
        )
    if tool == "trace_memory_relations":
        return trace_memory_relations(
            pg_config_path=pg_config_path,
            memory_id=str(request["memory_id"]),
            direction=str(request.get("direction") or "both"),
            max_depth=int(request.get("max_depth") or 1),
        )
    raise ValueError(f"unsupported tool: {tool}")


def _execute_tool_requests(
    requests: list[Mapping[str, Any]],
    *,
    source_database: Path,
    pg_config_path: Path,
    turn_id: str,
    seen_tool_keys: set[tuple[str, str]],
    max_chars: int,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for request in requests:
        key = _tool_key(request)
        if key in seen_tool_keys:
            observations.append(
                {
                    "status": "skipped",
                    "tool": key[0],
                    "target": key[1],
                    "reason": "same tool request already executed in this run",
                    "request": dict(request),
                }
            )
            continue
        seen_tool_keys.add(key)
        try:
            result = _execute_tool_request(
                request,
                source_database=source_database,
                pg_config_path=pg_config_path,
                turn_id=turn_id,
                max_chars=max_chars,
            )
            observations.append(
                {
                    "status": "complete",
                    "request": dict(request),
                    "result": result,
                }
            )
        except Exception as error:
            observations.append(
                {
                    "status": "failed",
                    "request": dict(request),
                    "error": str(error),
                }
            )
    return observations


def _update_turn_access_ids(
    available_ids: dict[str, set[str]],
    observations: list[dict[str, Any]],
) -> None:
    """Promote only successfully observed history into the authorization state."""

    discovered = available_ids.setdefault("discovered_turn_ids", set())
    observed = available_ids.setdefault("observed_turn_ids", set())
    evidence_ids = available_ids.setdefault("evidence_ids", set())
    pointer_ids = available_ids.setdefault("pointer_ids", set())
    historical_evidence = available_ids.setdefault("historical_evidence_ids", set())
    open_chain_scans = available_ids.setdefault("open_chain_scan_completed", set())
    open_chain_ids = available_ids.setdefault("open_chain_ids", set())
    for observation in observations:
        if observation.get("status") != "complete":
            continue
        result = observation.get("result")
        if not isinstance(result, Mapping) or result.get("status") != "complete":
            continue
        tool = result.get("tool")
        if tool == "list_open_memory_chains":
            open_chain_scans.add("current_session")
            current_turn_id = str(result.get("current_turn_id") or "")
            for chain in result.get("chains") or []:
                if not isinstance(chain, Mapping):
                    continue
                chain_id = str(chain.get("chain_id") or "")
                if chain_id:
                    open_chain_ids.add(chain_id)
                required_turn_ids = (
                    available_ids.setdefault(f"open_chain_required::{chain_id}", set())
                    if chain_id
                    else set()
                )
                for related_turn_id in chain.get("related_turn_ids") or []:
                    related_turn_id = str(related_turn_id or "")
                    if related_turn_id and related_turn_id != current_turn_id:
                        discovered.add(related_turn_id)
                        required_turn_ids.add(related_turn_id)
            continue
        if tool == "list_previous_session_turns":
            for item in result.get("turns") or []:
                if isinstance(item, Mapping) and item.get("turn_id"):
                    discovered.add(str(item["turn_id"]))
            continue
        if tool != "read_session_turn_evidence":
            continue
        target_turn_id = str(result.get("target_turn_id") or "")
        if not target_turn_id or target_turn_id not in discovered:
            continue
        detail = result.get("evidence")
        if not isinstance(detail, Mapping):
            continue
        turn = detail.get("turn")
        if not isinstance(turn, Mapping) or str(turn.get("turn_id") or "") != target_turn_id:
            continue
        observed.add(target_turn_id)
        layers = detail.get("layers")
        if not isinstance(layers, Mapping):
            continue
        for layer_name in ("core", "candidate", "pointer"):
            for item in layers.get(layer_name) or []:
                if not isinstance(item, Mapping):
                    continue
                evidence_id = str(item.get("evidence_id") or "")
                raw_pointer_id = str(item.get("raw_pointer_id") or "")
                if evidence_id:
                    evidence_ids.add(evidence_id)
                    historical_evidence.add(evidence_id)
                    if layer_name == "pointer":
                        pointer_ids.add(evidence_id)
                if raw_pointer_id:
                    pointer_ids.add(raw_pointer_id)


def _observed_memory_ids(observations: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for observation in observations:
        if observation.get("status") != "complete":
            continue
        tool_result = observation.get("result")
        if not isinstance(tool_result, Mapping):
            continue
        if tool_result.get("tool") == "read_memory_detail":
            memory = tool_result.get("memory")
            if isinstance(memory, Mapping) and memory.get("memory_id"):
                result.add(str(memory["memory_id"]))
        elif tool_result.get("tool") == "read_topic_detail":
            detail = tool_result.get("detail")
            if not isinstance(detail, Mapping):
                continue
            for memory in detail.get("memories") or []:
                if isinstance(memory, Mapping) and memory.get("memory_id"):
                    result.add(str(memory["memory_id"]))
    return result


def _auto_required_memory_reads(
    raw_decision: Mapping[str, Any],
    *,
    available_ids: Mapping[str, set[str]],
    observed_memory_ids: set[str],
) -> list[dict[str, Any]]:
    """Convert unsafe final modifications into required read-only lookups.

    This is a guardrail, not a write.  If the model tries to update/merge/mark
    conflict record on an existing memory before reading its body, the orchestrator
    forces a read_memory_detail observation and asks the model to decide again.
    """

    if raw_decision.get("decision") != "final_changes":
        return []
    changes = raw_decision.get("changes")
    if not isinstance(changes, Mapping):
        return []
    memory_ids = available_ids.get("memory_ids") or set()
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for change in changes.get("changes") or []:
        if not isinstance(change, Mapping):
            continue
        if change.get("action") not in {
            "merge_memory",
            "merge_existing_memories",
            "update_memory",
            "retire_memory",
            "move_memory_topic",
            "mark_conflict",
            "record_conflict_decision",
        }:
            continue
        target = str(change.get("target_memory_id") or "")
        if target and target in memory_ids and target not in observed_memory_ids and target not in seen:
            seen.add(target)
            result.append(
                {
                    "tool": "read_memory_detail",
                    "memory_id": target,
                    "reason": "程序强制补证据：修改、合并或冲突处理已有 memory 前必须先读取正文。",
                }
            )
        if change.get("action") == "merge_existing_memories":
            for source in change.get("source_memory_ids") or []:
                source = str(source)
                if not source or source not in memory_ids or source in observed_memory_ids or source in seen:
                    continue
                seen.add(source)
                result.append(
                    {
                        "tool": "read_memory_detail",
                        "memory_id": source,
                        "reason": "程序强制补证据：合并已有 source memory 前必须先读取正文。",
                    }
                )
    return result


def _auto_required_chain_reads(
    raw_decision: Mapping[str, Any],
    *,
    available_ids: Mapping[str, set[str]],
    max_requests: int = 3,
) -> list[dict[str, Any]]:
    """Run mandatory chain discovery and read missing chain evidence.

    A technical final decision is not valid until open chains were listed.
    When the model skips that protocol step, the orchestrator performs the
    read-only discovery call and asks the model to decide again.  If the model
    then selects an existing chain from that result, all unread chain turns are
    loaded in the same way.  These reads add evidence; they never choose a
    state or write a memory on the model's behalf.
    """

    if raw_decision.get("decision") != "final_changes":
        return []
    chain_states: list[Mapping[str, Any]] = []
    chain_state = raw_decision.get("chain_state")
    if isinstance(chain_state, Mapping):
        chain_states.append(chain_state)
    raw_operations = raw_decision.get("chain_operations")
    if isinstance(raw_operations, list):
        chain_states.extend(
            operation.get("chain_state")
            for operation in raw_operations
            if isinstance(operation, Mapping)
            and isinstance(operation.get("chain_state"), Mapping)
        )
    if not chain_states:
        return []
    if (
        any(
            str(state.get("chain_type") or "") in {"compilation", "runtime"}
            for state in chain_states
        )
        and "current_session"
        not in (available_ids.get("open_chain_scan_completed") or set())
        and not (available_ids.get("open_chain_ids") or set())
    ):
        return [
            {
                "tool": "list_open_memory_chains",
                "reason": (
                    "程序强制补协议步骤：技术最终判断前必须先查看同会话未闭环故障链。"
                ),
                "limit": 10,
            }
        ]
    observed_turn_ids = available_ids.get("observed_turn_ids") or set()
    missing_turn_ids: set[str] = set()
    for state in chain_states:
        chain_id = str(state.get("chain_id") or "")
        if not chain_id or chain_id not in (available_ids.get("open_chain_ids") or set()):
            continue
        required_turn_ids = available_ids.get(f"open_chain_required::{chain_id}") or set()
        missing_turn_ids.update(set(required_turn_ids) - set(observed_turn_ids))
    return [
        {
            "tool": "read_session_turn_evidence",
            "turn_id": turn_id,
            "reason": (
                "程序强制补证据：继续既有故障链前，必须读取该链关联历史 turn 的详细证据。"
            ),
            "layers": ["core", "candidate", "pointer"],
            "max_chars": 4000,
        }
        for turn_id in sorted(missing_turn_ids)[:max_requests]
    ]


def _react_payload(
    *,
    initial_context: Mapping[str, Any],
    observations: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
    available_ids: Mapping[str, set[str]],
) -> dict[str, Any]:
    observed_memory_ids = _observed_memory_ids(observations)
    all_memory_ids = set(available_ids.get("memory_ids") or set())
    summary_only_memory_ids = all_memory_ids - observed_memory_ids
    return {
        "schema_version": "memory-react-input-v1",
        "workflow_stage": "react_memory_maintenance",
        "iteration": iteration,
        "max_iterations": max_iterations,
        "memory_policy": {
            "role": "长期记忆整理 Agent，不是普通总结器",
            "long_term_value_rule": "只保存以后能减少重复问答、减少错误、减少返工的信息",
            "admission_order": [
                "提取有证据支持的事实，不按整轮主题一刀切。",
                "去掉项目名、路径和具体数值，抽象因果机制或协作规则。",
                "抽象后仍能指导相似任务才有长期价值。",
                "按 memory_type 检查必备证据和验证状态。",
            ],
            "primary_targets": [
                "用户交互习惯和协作偏好",
                "编译成功/失败经验",
                "用户反馈运行现象后定位出的原因和解决方法",
            ],
            "reject_if": "去掉当前项目细节后不能留下独立、可执行的机制或协作规则。",
        },
        "memory_access_state": {
            "summary_only_memory_ids": sorted(summary_only_memory_ids),
            "observed_memory_ids": sorted(observed_memory_ids),
            "modifiable_memory_ids": sorted(observed_memory_ids),
            "rule": "memory_index 只是摘要索引；只有 modifiable_memory_ids 中的 memory 才允许 update/merge/merge_existing/retire/move_memory_topic/mark_conflict/record_conflict_decision。",
        },
        "turn_access_state": {
            "discovered_turn_ids": sorted(available_ids.get("discovered_turn_ids") or set()),
            "observed_turn_ids": sorted(available_ids.get("observed_turn_ids") or set()),
            "authorized_historical_evidence_ids": sorted(
                available_ids.get("historical_evidence_ids") or set()
            ),
            "rule": "历史摘要只用于发现候选；只有 observed_turn_ids 中已读取详情的 turn，其返回 evidence_id 才可在后续跨 turn 提交阶段使用。",
        },
        "action_decision_guide": {
            "create_memory": "有长期价值，已有 memory 不能覆盖，能形成清晰 stable_key，证据足够。",
            "merge_memory": "同一模式，只增加证据或例子，不改变原结论；必须先读过目标 memory 详情。",
            "merge_existing_memories": "多条 active memory 语义重复或同方向，继续分开会增加召回噪声；必须先读过 target 和 sources 的详情。",
            "update_memory": "已有 memory 方向正确，本轮提供更准确、更通用或更完整表述；必须先读过目标 memory 详情。",
            "retire_memory": "已有 active memory 已失效、重复到不应继续作为当前有效记忆或被明确废弃；必须先读过目标 memory 详情，不能修改 topic。",
            "move_memory_topic": "memory 内容正确但归属主题错误时移动到更准确的 active topic；必须先读过目标 memory 详情，不能修改 memory 内容或 topic 摘要。",
            "merge_topics": "多个 active topic 语义重复时合并主题目录；只迁移 source topic 下 active memories 并标记 source topic merged，不修改 memory 内容或 target topic 摘要。",
            "mark_conflict": "同 scope、同条件、同对象下结论相反且无法用环境差异解释；必须先读过目标 memory 详情。",
            "update_topic": "memory 内容和 topic 标题/概要出现语义漂移时维护主题元信息；若要判断主题是否应改名，优先先读过 topic 详情。",
            "record_conflict_decision": "记录同一作用域前后矛盾的旧说法、新说法、处理决定和原因；必须先读过目标 memory 详情。",
            "noop": "一次性参数、低价值信息、已有记忆完全覆盖且无新增证据、或未验证猜测。",
            "request_more_evidence": "摘要不足或需要原始片段/后续验证才能判断。",
        },
        "limits": {
            "max_tool_requests_per_iteration": 3,
            "tool_result_max_chars": 4000,
            "final_changes_schema": MEMORY_CHANGES_SCHEMA_VERSION,
        },
        "available_tools": [
            {
                "tool": "search_related_memories",
                "args": {
                    "query": "必填；来自当前 turn 证据或已有索引的真实关键词",
                    "memory_type": "可选；只能是 user_preference / compile_failure_experience / runtime_failure_experience",
                    "domain": "可选；限制领域 scope",
                    "project_key": "可选；限制项目 scope",
                    "limit": "可选；1-20，默认 8",
                },
                "purpose": "当当前 memory_index 摘要不足或怀疑旧记忆未出现在概览中时，搜索主题标题/摘要和 memory 摘要/正文，返回候选 memory_id/topic_id；搜索结果不能直接授权修改",
            },
            {
                "tool": "read_topic_detail",
                "args": {"topic_id": "必须来自 available_ids.topic_ids"},
                "purpose": "读取主题正文和主题下的 active memories，用于判断合并/更新/冲突/重复",
            },
            {
                "tool": "list_topic_memories",
                "args": {
                    "topic_id": "必须来自 available_ids.topic_ids",
                    "limit": "可选；1-50，默认 20",
                },
                "purpose": "轻量列出某主题下的 memory 摘要，用于判断是否已有重复/相似 memory；如需修改仍必须 read_memory_detail",
            },
            {
                "tool": "read_memory_detail",
                "args": {"memory_id": "必须来自 available_ids.memory_ids"},
                "purpose": "读取某条 memory 的完整内容，用于判断是否合并、更新或冲突",
            },
            {
                "tool": "read_memory_history",
                "args": {
                    "memory_id": "必须来自 available_ids.memory_ids",
                    "limit": "可选；1-50，默认 10",
                },
                "purpose": "读取 memory 的版本历史，用于判断新证据是补充、修正、已处理过，还是需要冲突记录；不授权直接修改",
            },
            {
                "tool": "list_memory_conflicts",
                "args": {
                    "memory_id": "可选；必须来自 available_ids.memory_ids",
                    "topic_id": "可选；必须来自 available_ids.topic_ids；memory_id/topic_id 二选一",
                    "limit": "可选；1-50，默认 20",
                },
                "purpose": "查看某条 memory 或某个 topic 下已有冲突记录，避免重复记录冲突，或判断是否需要补充冲突证据",
            },
            {
                "tool": "expand_pointer",
                "args": {"raw_pointer_id": "必须来自本轮 pointer id"},
                "purpose": "从 CP SQLite 回查原始片段，补足 core/candidate 摘要不足的信息",
            },
            {
                "tool": "get_current_turn_evidence_summary",
                "args": {
                    "max_items_per_layer": "可选；1-50，默认 12",
                    "max_chars": "可选；80-4000，默认使用工具结果上限",
                },
                "purpose": "回看当前 turn 的 core/candidate/pointer 紧凑摘要和可引用 evidence_id；不能直接授权修改已有 memory",
            },
            {
                "tool": "list_open_memory_chains",
                "args": {"limit": "可选；1-20，默认 10"},
                "purpose": "列出同用户、同数据源、同会话、同项目的未闭环编译/上板链及关联 turn 摘要；只用于发现，不直接授权历史 evidence_id",
            },
            {
                "tool": "list_previous_session_turns",
                "args": {
                    "limit": "可选；1-20，默认 10",
                    "preview_chars": "可选；80-800，默认 320",
                },
                "purpose": "只列出同用户、同数据源、同会话、同项目且早于当前 turn 的摘要；只用于发现相关历史 turn",
            },
            {
                "tool": "read_session_turn_evidence",
                "args": {
                    "turn_id": "必须来自已完成的 list_previous_session_turns 结果",
                    "layers": "可选；core/candidate/pointer 的非重复子集",
                    "max_chars": "可选；80-8000",
                },
                "purpose": "读取一个已发现历史 turn 的三层证据；成功后才授权该 turn 和返回 evidence_id",
            },
            {
                "tool": "trace_memory_relations",
                "args": {
                    "memory_id": "必须来自 available_ids.memory_ids",
                    "direction": "可选；backward/forward/both，默认 both",
                    "max_depth": "可选；1-5，默认 1",
                },
                "purpose": "查看某条 memory 的版本历史和 supersedes/merges/conflicts_with 关系；用于判断来源、去向、合并或冲突背景",
            },
        ],
        "available_ids": _ids_for_log(available_ids),
        "initial_context": initial_context,
        "observations": observations,
        "required_output_schema": {
            "schema_version": MEMORY_REACT_SCHEMA_VERSION,
            "decision": "tool_requests | final_changes",
            "reason": "中文原因",
            "chain_state": {
                "required_for": "single-chain compilation/runtime final decisions",
                "optional_for_non_technical": True,
                "fields": ["chain_id?", "chain_type", "state", "related_turn_ids", "missing_evidence"],
                "state_enums": {
                    "none": ["not_applicable"],
                    "compilation": ["failure_observed", "cause_identified", "fix_applied", "fix_rejected", "compile_verified"],
                    "runtime": ["symptom_observed", "cause_identified", "fix_applied", "waiting_user_validation", "user_verified", "fix_rejected"],
                },
                "missing_evidence_rule": "open technical state requires non-empty array; closed or none state requires []",
                "chain_id_rule": "only copy an existing chain_id returned by list_open_memory_chains",
            },
            "chain_operations": {
                "required_for": "a final decision that mutates two or more independent technical chains",
                "mutually_exclusive_with": "chain_state",
                "max_items": 4,
                "fields": ["operation_id", "mode", "chain_state", "linked_change_indexes"],
                "mode_enums": ["update_existing", "create_new"],
                "operation_id_rule": "unique op_[a-z0-9_] identifier within this decision",
                "link_rule": "every technical memory change index is linked by exactly one closed operation; open operations use []",
            },
            "tool_requests": [],
            "changes": "[] when tool_requests; memory-changes-v1 object when final_changes",
        },
    }


def _validate_or_repair_react(
    *,
    model: OpenAICompatibleJsonClient,
    first: Mapping[str, Any],
    payload: Mapping[str, Any],
    turn_id: str,
    available_ids: Mapping[str, set[str]],
    observed_memory_ids: set[str],
    pg_config_path: Path,
) -> tuple[dict[str, Any] | None, bool, list[str], Mapping[str, Any]]:
    try:
        return (
            validate_react_decision(
                first,
                turn_id=turn_id,
                available_ids=available_ids,
                observed_memory_ids=observed_memory_ids,
                pg_config_path=pg_config_path,
            ),
            False,
            [],
            first,
        )
    except MemoryReactValidationError as error:
        previous: Mapping[str, Any] = first
        errors = error.errors
        for repair_attempt in range(1, 3):
            repair_payload = {
                "repair_attempt": repair_attempt,
                "previous_output": previous,
                "validation_errors": errors,
                "available_ids": _ids_for_log(available_ids),
                "observed_memory_ids": sorted(observed_memory_ids),
                "required_schema": payload["required_output_schema"],
                "current_turn_id": turn_id,
                "strict_field_rules": {
                    "memory_changes_top_level": [
                        "schema_version",
                        "turn_id",
                        "source_turn_ids?",
                        "reason",
                        "changes",
                    ],
                    "allowed_create_memory_types": sorted(ALLOWED_MEMORY_TYPES),
                    "create_memory": [
                        "action", "reason", "client_temp_id?", "topic|topic_id",
                        "memory", "technical_closure?", "evidence_refs",
                    ],
                    "noop": ["action", "reason", "evidence_refs"],
                    "request_more_evidence": ["action", "reason", "requests", "evidence_refs"],
                    "merge_memory": ["action", "reason", "target_memory_id", "memory", "technical_closure?", "evidence_refs"],
                    "merge_existing_memories": [
                        "action",
                        "reason",
                        "target_memory_id",
                        "source_memory_ids",
                        "memory",
                        "technical_closure?",
                        "evidence_refs",
                        "conflict_disposition",
                    ],
                    "update_memory": ["action", "reason", "target_memory_id", "memory", "technical_closure?", "evidence_refs"],
                    "retire_memory": ["action", "reason", "target_memory_id", "evidence_refs", "conflict_disposition"],
                    "move_memory_topic": ["action", "reason", "target_memory_id", "from_topic_id", "to_topic_id", "evidence_refs"],
                    "merge_topics": ["action", "reason", "target_topic_id", "source_topic_ids", "evidence_refs"],
                    "mark_conflict": ["action", "reason", "target_memory_id", "memory", "evidence_refs"],
                    "update_topic": ["action", "reason", "target_topic_id", "topic", "evidence_refs"],
                    "record_conflict_decision": ["action", "reason", "target_memory_id", "conflict", "evidence_refs"],
                    "memory_for_existing_target": ["summary", "content", "confidence"],
                    "technical_closure": ["chain_type", "state", "verification_evidence_ids"],
                    "topic_patch": ["title", "summary"],
                    "conflict": ["conflict_type", "old_claim", "new_claim", "decision", "reason", "status", "metadata"],
                    "tool_request_expand_pointer": ["tool", "raw_pointer_id", "reason"],
                    "tool_request_read_memory_detail": ["tool", "memory_id", "reason"],
                    "tool_request_read_topic_detail": ["tool", "topic_id", "reason"],
                    "tool_request_list_topic_memories": ["tool", "topic_id", "reason", "limit?"],
                    "tool_request_read_memory_history": ["tool", "memory_id", "reason", "limit?"],
                    "tool_request_list_memory_conflicts": [
                        "tool",
                        "memory_id?",
                        "topic_id?",
                        "reason",
                        "limit?",
                    ],
                    "tool_request_search_related_memories": [
                        "tool",
                        "query",
                        "reason",
                        "memory_type?",
                        "domain?",
                        "project_key?",
                        "limit?",
                    ],
                    "tool_request_list_open_memory_chains": [
                        "tool",
                        "reason",
                        "limit?",
                    ],
                    "tool_request_list_previous_session_turns": [
                        "tool",
                        "reason",
                        "limit?",
                        "preview_chars?",
                    ],
                    "tool_request_read_session_turn_evidence": [
                        "tool",
                        "turn_id",
                        "reason",
                        "layers?",
                        "max_chars?",
                    ],
                },
            }
            repaired_output = model.complete_json(
                system_prompt=REACT_REPAIR_PROMPT,
                user_payload=repair_payload,
                max_tokens=7000,
            )
            try:
                return (
                    validate_react_decision(
                        repaired_output,
                        turn_id=turn_id,
                        available_ids=available_ids,
                        observed_memory_ids=observed_memory_ids,
                        pg_config_path=pg_config_path,
                    ),
                    True,
                    [],
                    repaired_output,
                )
            except MemoryReactValidationError as repaired_error:
                previous = repaired_output
                errors = repaired_error.errors
        return None, True, errors, previous


def run_memory_react_agent(
    *,
    source_database: Path,
    turn_id: str,
    pg_config_path: Path = DEFAULT_PG_CONFIG,
    config_path: Path = DEFAULT_CONFIG,
    memory_limit: int = 80,
    topic_limit: int = 200,
    dry_run: bool = False,
    max_iterations: int = 3,
    log_dir: Path = DEFAULT_REACT_RUN_DIR,
    client: OpenAICompatibleJsonClient | None = None,
) -> dict[str, Any]:
    max_iterations = max(1, min(int(max_iterations), 10))
    run_id = _safe_run_id(turn_id).replace("agent-", "react-", 1)
    started_at = _utc_now_iso()
    steps: list[dict[str, Any]] = []
    initial_context = build_memory_agent_input(
        turn_id=turn_id,
        pg_config_path=pg_config_path,
        memory_limit=memory_limit,
        topic_limit=topic_limit,
    )
    if initial_context.get("status") != "complete":
        steps.append(_step("build_react_input", "failed", f"turn unavailable: {turn_id}"))
        result = {
            "status": "failed",
            "stage": "input",
            "turn_id": turn_id,
            "committed": False,
            "input": initial_context,
        }
        run_log = {
            "schema_version": "memory-react-run-log-v1",
            "run_id": run_id,
            "turn_id": turn_id,
            "mode": "dry_run" if dry_run else "commit",
            "started_at": started_at,
            "finished_at": _utc_now_iso(),
            "status": result["status"],
            "steps": steps,
            "result": result,
        }
        return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}

    stats = _context_stats(initial_context)
    available_ids = _collect_available_ids(initial_context)
    steps.extend(
        [
            _step(
                "build_react_input",
                "complete",
                f"core={stats['core_count']} candidate={stats['candidate_count']} pointer={stats['pointer_count']}",
                filters=initial_context.get("filters"),
            ),
            _step("load_memory_index", "complete", f"topics={stats['topic_count']} memories={stats['memory_count']}"),
        ]
    )

    model = client or OpenAICompatibleJsonClient.from_config(config_path)
    observations: list[dict[str, Any]] = []
    iteration_logs: list[dict[str, Any]] = []
    seen_tool_keys: set[tuple[str, str]] = set()
    final_decision: dict[str, Any] | None = None
    raw_final: Mapping[str, Any] | None = None

    for iteration in range(1, max_iterations + 1):
        payload = _react_payload(
            initial_context=initial_context,
            observations=observations,
            iteration=iteration,
            max_iterations=max_iterations,
            available_ids=available_ids,
        )
        raw = model.complete_json(
            system_prompt=REACT_SYSTEM_PROMPT,
            user_payload=payload,
            max_tokens=9000,
        )
        auto_chain_reads = _auto_required_chain_reads(
            raw,
            available_ids=available_ids,
            max_requests=3,
        )
        auto_memory_reads = _auto_required_memory_reads(
            raw,
            available_ids=available_ids,
            observed_memory_ids=_observed_memory_ids(observations),
        )
        auto_reads = [*auto_chain_reads, *auto_memory_reads][:3]
        if auto_reads:
            new_observations = _execute_tool_requests(
                auto_reads,
                source_database=source_database,
                pg_config_path=pg_config_path,
                turn_id=turn_id,
                seen_tool_keys=seen_tool_keys,
                max_chars=4000,
            )
            observations.extend(new_observations)
            _update_turn_access_ids(available_ids, new_observations)
            iteration_logs.append(
                {
                    "iteration": iteration,
                    "decision": "auto_required_reads",
                    "repaired": False,
                    "raw": raw,
                    "auto_tool_requests": auto_reads,
                }
            )
            steps.append(
                _step(
                    "auto_required_reads",
                    "complete",
                    f"iteration={iteration} requests={len(auto_reads)}",
                )
            )
            continue
        decision, repaired, errors, raw_after_validation = _validate_or_repair_react(
            model=model,
            first=raw,
            payload=payload,
            turn_id=turn_id,
            available_ids=available_ids,
            observed_memory_ids=_observed_memory_ids(observations),
            pg_config_path=pg_config_path,
        )
        iteration_log: dict[str, Any] = {
            "iteration": iteration,
            "repaired": repaired,
            "raw": raw_after_validation,
        }
        if decision is None:
            steps.append(
                _step(
                    "validate_react_decision",
                    "failed",
                    f"iteration={iteration} errors={len(errors)}",
                    errors=errors,
                )
            )
            result = {
                "status": "failed",
                "stage": "validate_react_decision",
                "turn_id": turn_id,
                "iteration": iteration,
                "validation_errors": errors,
                "raw_decision": raw_after_validation,
                "committed": False,
            }
            run_log = {
                "schema_version": "memory-react-run-log-v1",
                "run_id": run_id,
                "turn_id": turn_id,
                "mode": "dry_run" if dry_run else "commit",
                "started_at": started_at,
                "finished_at": _utc_now_iso(),
                "status": result["status"],
                "steps": steps,
                "iterations": [*iteration_logs, iteration_log],
                "observations": observations,
                "result": result,
            }
            return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}

        iteration_log["decision"] = decision.get("decision")
        iteration_logs.append(iteration_log)
        if decision.get("decision") == "tool_requests":
            requests = [dict(item) for item in decision.get("tool_requests") or []]
            new_observations = _execute_tool_requests(
                requests,
                source_database=source_database,
                pg_config_path=pg_config_path,
                turn_id=turn_id,
                seen_tool_keys=seen_tool_keys,
                max_chars=4000,
            )
            observations.extend(new_observations)
            _update_turn_access_ids(available_ids, new_observations)
            steps.append(
                _step(
                    "execute_read_tools",
                    "complete",
                    f"iteration={iteration} requests={len(requests)} observations={len(new_observations)}",
                )
            )
            continue

        final_decision = decision
        raw_final = raw_after_validation
        break

    if final_decision is None:
        result = {
            "status": "failed",
            "stage": "max_iterations",
            "turn_id": turn_id,
            "iterations": max_iterations,
            "observation_count": len(observations),
            "committed": False,
        }
        steps.append(_step("react_loop", "failed", "max iterations reached without final_changes"))
        run_log = {
            "schema_version": "memory-react-run-log-v1",
            "run_id": run_id,
            "turn_id": turn_id,
            "mode": "dry_run" if dry_run else "commit",
            "started_at": started_at,
            "finished_at": _utc_now_iso(),
            "status": result["status"],
            "steps": steps,
            "iterations": iteration_logs,
            "observations": observations,
            "result": result,
        }
        return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}

    plan = dict(final_decision["changes"])
    chain_state = final_decision.get("chain_state")
    chain_operations = [
        dict(item)
        for item in final_decision.get("chain_operations") or []
        if isinstance(item, Mapping)
    ]
    change_count = len(plan.get("changes") or [])
    steps.append(_step("final_changes", "complete", f"changes={change_count}"))
    if isinstance(chain_state, Mapping):
        steps.append(
            _step(
                "reconstruct_chain_state",
                "complete",
                f"chain_type={chain_state.get('chain_type')} state={chain_state.get('state')}",
                related_turn_ids=list(chain_state.get("related_turn_ids") or []),
                missing_evidence=list(chain_state.get("missing_evidence") or []),
            )
        )
    for operation in chain_operations:
        operation_state = operation.get("chain_state") or {}
        steps.append(
            _step(
                "reconstruct_chain_operation",
                "complete",
                (
                    f"operation_id={operation.get('operation_id')} mode={operation.get('mode')} "
                    f"chain_type={operation_state.get('chain_type')} state={operation_state.get('state')}"
                ),
                linked_change_indexes=list(operation.get("linked_change_indexes") or []),
                related_turn_ids=list(operation_state.get("related_turn_ids") or []),
                missing_evidence=list(operation_state.get("missing_evidence") or []),
            )
        )

    if dry_run:
        result = {
            "status": "complete",
            "mode": "dry_run",
            "turn_id": turn_id,
            "iterations": len(iteration_logs),
            "observation_count": len(observations),
            "chain_state": chain_state,
            "chain_operations": chain_operations,
            "plan": plan,
            "committed": False,
        }
        run_log = {
            "schema_version": "memory-react-run-log-v1",
            "run_id": run_id,
            "turn_id": turn_id,
            "mode": "dry_run",
            "started_at": started_at,
            "finished_at": _utc_now_iso(),
            "status": result["status"],
            "steps": steps,
            "iterations": iteration_logs,
            "observations": observations,
            "raw_final_decision": raw_final,
            "result": result,
        }
        return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}

    commit = commit_memory_changes(plan, pg_config_path=pg_config_path)
    committed = bool((commit.get("counts") or {}).get("applied"))
    steps.append(
        _step(
            "commit_memory_changes",
            "complete",
            f"applied={(commit.get('counts') or {}).get('applied', 0)} rejected={(commit.get('counts') or {}).get('rejected', 0)}",
        )
    )
    plan_changes = plan.get("changes") if isinstance(plan.get("changes"), list) else []
    runtime_operations = list(chain_operations)
    legacy_operation = False
    if not runtime_operations and isinstance(chain_state, Mapping) and chain_state.get("chain_type") != "none":
        legacy_operation = True
        linked_indexes = [
            index
            for index, change in enumerate(plan_changes)
            if isinstance(change, Mapping)
            and isinstance(change.get("technical_closure"), Mapping)
            and change["technical_closure"].get("chain_type") == chain_state.get("chain_type")
            and change["technical_closure"].get("state") == chain_state.get("state")
            and (chain_state.get("chain_type"), chain_state.get("state")) in CLOSED_STATES
        ]
        runtime_operations = [
            {
                "operation_id": "op_legacy_single",
                "mode": "legacy_auto",
                "chain_state": dict(chain_state),
                "linked_change_indexes": linked_indexes,
            }
        ]

    applied_memory_ids: dict[int, list[str]] = {}
    for item in commit.get("applied") or []:
        if not isinstance(item, Mapping):
            continue
        index = item.get("index")
        if not isinstance(index, int) and len(plan_changes) == 1:
            index = 0
        if not isinstance(index, int) or index < 0 or index >= len(plan_changes):
            continue
        memory_id = item.get("memory_id") or item.get("target_memory_id")
        if memory_id and str(memory_id) not in applied_memory_ids.setdefault(index, []):
            applied_memory_ids[index].append(str(memory_id))

    chain_persistences: list[dict[str, Any]] = []
    for operation in runtime_operations:
        operation_id = str(operation.get("operation_id") or "")
        mode = str(operation.get("mode") or "")
        operation_state = operation.get("chain_state") or {}
        if commit.get("ok"):
            final_memory_ids = [
                memory_id
                for index in operation.get("linked_change_indexes") or []
                for memory_id in applied_memory_ids.get(index, [])
            ]
            persistence = save_memory_chain_state(
                pg_config_path=pg_config_path,
                current_turn_id=turn_id,
                chain_state=operation_state,
                final_memory_ids=final_memory_ids,
                create_new=mode == "create_new",
                creation_key=operation_id if mode == "create_new" else "",
            )
        else:
            persistence = {
                "status": "skipped",
                "result": "memory_commit_failed",
                "guidance": "记忆提交存在拒绝项，故障链状态未推进。",
            }
        persistence_with_operation = {
            "operation_id": operation_id,
            "mode": mode,
            **persistence,
        }
        chain_persistences.append(persistence_with_operation)
        chain_status = str(persistence.get("status") or "unknown")
        steps.append(
            _step(
                "persist_chain_state" if legacy_operation else "persist_chain_operation",
                "complete" if chain_status in {"complete", "skipped"} else "failed",
                (
                    f"operation_id={operation_id} mode={mode} status={chain_status} "
                    f"result={persistence.get('result', '')}"
                ),
                chain_id=(persistence.get("chain") or {}).get("chain_id"),
                error_code=persistence.get("error_code"),
            )
        )
    chain_persistence = (
        {key: value for key, value in chain_persistences[0].items() if key not in {"operation_id", "mode"}}
        if legacy_operation and chain_persistences
        else None
    )
    result_status = "complete"
    if not commit.get("ok") or any(
        item.get("status") in {"rejected", "denied", "retry", "not_found"}
        for item in chain_persistences
    ):
        result_status = "partial_failure"
    result = {
        "status": result_status,
        "mode": "commit",
        "turn_id": turn_id,
        "iterations": len(iteration_logs),
        "observation_count": len(observations),
        "chain_state": chain_state,
        "chain_operations": chain_operations,
        "plan": plan,
        "commit": commit,
        "chain_persistence": chain_persistence,
        "chain_persistences": chain_persistences,
        "committed": committed,
    }
    run_log = {
        "schema_version": "memory-react-run-log-v1",
        "run_id": run_id,
        "turn_id": turn_id,
        "mode": "commit",
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "status": result["status"],
        "steps": steps,
        "iterations": iteration_logs,
        "observations": observations,
        "raw_final_decision": raw_final,
        "result": result,
    }
    return {**result, "run_id": run_id, "log": _write_run_log(run_log=run_log, log_dir=log_dir)}


__all__ = [
    "MEMORY_REACT_SCHEMA_VERSION",
    "MemoryReactValidationError",
    "run_memory_react_agent",
    "validate_react_decision",
]
