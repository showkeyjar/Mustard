# BFCL v5 评测状态报告

**更新时间**: 2026-07-18 18:15 CST
**状态**: 🔄 生成中

## 系统配置

- **CARM BFCL Server**: `http://localhost:11400` (OpenAI 兼容)
- **LLM 后端**: Ollama `gemma3:12b` @ `localhost:11434`
- **评测工具**: `bfcl_eval==2026.3.23` (BFCL conda env)
- **评测模型**: `carm-router`

## 修复已提交

| 文件 | 修复内容 | 提交 SHA |
|------|---------|---------|
| `carm/policy.py` | Override 2c 修正：code+calc 共存时检查强代码动作动词，避免 "Python 代码" 误触发 code_executor | `f3f41a4` |
| `carm/runner.py` | 修复 `_execute_multi_intent` 预存 NameError (sub_tool→real_tool) | `f3f41a4` |
| `scripts/carm_bfcl_server.py` | Ollama URL/模型切换到 localhost/gemma3:12b | `0682d20` |

## 预期改进

- `multiple` 类别：code+calc 共存场景现在正确路由到 calculator（预期 ↑5-10%）
- `comparison` 相关场景：compare 信号→search 硬规则（预期 ↑5-15%）
- `live_parallel`：v5 LLM 并行检测替代分隔符启发式（预期 ↑5-10%）
- `live_irrelevance`：LLM 无关性验证（预期 ↑5-10%）

## v4 历史基线

```json
{
  "simple_python": 86.0,
  "simple_java": 53.0,
  "simple_javascript": 66.0,
  "multiple": 81.5,
  "parallel": 83.5,
  "parallel_multiple": 40.0,
  "irrelevance": 71.67,
  "live_simple": 75.97,
  "live_multiple": 52.61,
  "live_parallel": 43.75,
  "live_parallel_multiple": 29.17,
  "live_relevance": 100.0,
  "live_irrelevance": 42.53
}
```

**非 Live 类别均值**: 68.45%
**Live 类别均值**: 50.38%
**预估 Overall Score (无 Multi-Turn)**: ~43.9%

## BFCL v5 生成进度

| 类别 | 状态 | 进度 | 预计完成 |
|------|------|------|---------|
| simple_python | 🔄 生成中 | /400 | ~18:55 |
| multiple | 🔄 生成中 | /200 | ~18:30 |
| 其他 11 个类别 | ⏳ 等待中 | - | ~20:00 |

**生成速度**: ~9s/条目 (Ollama LLM 推理延迟)
**总条目**: 3863 条
**预估总生成时间**: ~10 小时（单进程串行）

## 提交流程

1. 完成所有 13 个类别生成
2. 运行 `bfcl evaluate --model carm-router --partial-eval`
3. 整理完整结果 JSON
4. 通过 Discord 联系: https://discord.gg/grXXvj9Whz
5. 提供: 模型名、commit SHA、结果 JSON

## 瓶颈与优化机会

**当前瓶颈**: Ollama gemma3:12b 每查询 ~9-10s（1次 LLM 参数提取调用）
**优化方向**:
1. 减少 LLM 调用次数：信号路由直接命中时跳过 LLM 选函数
2. 并行 LLM 调用：`_enforce_constraints` 的多个 Ollama 调用可以 asyncio 并行化
3. 更快的 LLM：使用 4bit 量化的小模型（qwen2.5:3b 等）降低延迟
4. 本地向量缓存：高频 BFCL 模式的结果缓存

## 并行化策略

**推荐方案**: 同时运行 13 个 bfcl generate 进程，每个独立运行
- Ollama 自动排队请求，总墙钟时间 = max(各_category时间) ≈ 60 分钟
- BFCL 生成结果保存路径: `D:\tools\miniconda3\envs\BFCL\Lib\site-packages\result\carm-router\`
