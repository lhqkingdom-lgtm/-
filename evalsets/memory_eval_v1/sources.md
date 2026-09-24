# memory_eval_v1 数据来源

## 原始库

```text
D:\opencode-dev\opencode-dev.db
```

## 隔离索引库

```text
D:\codepulse\Codepulse-Memory-Lab\data\eval\memory_eval_index.db
```

这个隔离库只用于挑选与回放测评样本，不作为 live 轮询库使用。

## 已回填同事真实会话

| session_id | 标题 | 目录 | 说明 |
|---|---|---|---|
| `ses_f4c8c4db0ffetRhO3IjKePk27o` | 需求书与IO配置项目开发 | `C:\Users\miji-intern\Proj\TG035_918` | 完整需求开发、烧录反馈、FLASH 容量、LED/定时问题 |
| `ses_f528b583effe0gqY2ML7I1LWKf` | 需求书与IO配置产品代码生成 | `C:\Users\miji-intern\Proj\TG035_917` | 多轮 LED 映射、极性、运行反馈 |
| `ses_f7c1f7464ffeu0WsT5QPGx2oP0` | EL48项目IO配置代码生成 | `C:\Users\miji-intern\Proj\EL48` | 感应功能、调光闪烁、ADC 通道偏移、报告偏好 |
| `ses_f5c178495ffeLN1uUhRaowtaPr` | 基于FL20Pro需求生成完整项目代码 | `C:\Users\miji-intern\Proj\FL20PRO_2` | 多轮烧录反馈、标准源码对比、功能缺失定位 |
| `ses_f580bcec1ffeB0CLjWDzsdFSu0` | 基于需求书与IO配置的项目代码实现 | `C:\Users\miji-intern\Proj\FL20PRO_3` | 编译/云编译、运行现象、工具能力纠正 |
| `ses_f57fc71dfffeAtQSUMc15JOj49` | 需求书与IO引脚配置项目开发 | `C:\Users\miji-intern\Proj\T55_2` | continue/中断边界、芯片支持边界 |
| `ses_f5bfc4399ffeMzf42mpQfg6ohJ` | TG035源码对比报告及功能缺失分析 | `C:\Users\miji-intern\Proj\TG035-1.0.85` | 生产级源码对比、按键映射分析 |

## 样本规模

- 回填用户事件：85 条
- 固化单 turn case：34 条
- 固化连续 scenario：7 组
