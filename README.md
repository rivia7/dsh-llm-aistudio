# dsh-llm-aistudio

DeepSeek Harness（dsh）的 Google AI Studio 模型提供方插件：用你自己的 Gemini API key，直接调用 `generativelanguage.googleapis.com` 的 REST 流式接口。**不依赖 pi-ai，也不依赖 @google/genai**，请求转换、流解析和所有 finishReason 的处理都在本包里，所以新模型、新 finish reason 都不需要等上游升级依赖。

默认提供 `aistudio` 路由，内置两个模型：

| 模型 | 上下文 | 最大输出 | 推理强度 |
|---|---|---|---|
| `gemini-4-carbon` | 1,048,576 | 32,768 | low / medium / high（thinkingLevel） |
| `gemini-4-argon` | 1,048,576 | 32,768 | low / medium / high（thinkingLevel） |

## 安装与更新

```sh
sh scripts/install.sh web     # 打包并装进 web profile，然后重启 dsh web
```

脚本会 `npm pack` 到 `dist/`，再执行 `dsh plugin --profile web add file:…tgz`。因为本包声明了 `dsh.bundle`，`dsh plugin add` 会自动把它加进 profile 的 `dsh.profile.bundles`，不需要手动挂载。修改代码后提升 `package.json` 的版本号，再跑一次脚本即可。

## API key

插件按 `apiKeyEnv`（默认 `GEMINI_API_KEY`）解析 key：先查 dsh 凭据库 `~/.dsh/.credentials.yaml` 的 `refs`，再查启动环境、项目 `.env` 和 `~/.dsh/.env`。凭据库里的值优先于环境变量；换 key 时要更新凭据库（web 设置页或直接编辑文件），改 `~/.zshrc` 不会生效。

## 配置

在 profile 的 `cordis.patch.yml` 里写一个指向 `id: llm-aistudio` 的条目即可覆盖默认值。例如加一个 Gemini 3.8 Flash，并让 Gemini 2.5 Flash 用 token 预算控制推理：

```yaml
- id: llm-aistudio
  name: dsh-llm-aistudio
  config:
    models:
      - id: gemini-4-carbon
        name: Gemini 4 Carbon
        contextWindow: 1048576
        maxTokens: 32768
        efforts: { low: LOW, medium: MEDIUM, high: HIGH }
      - id: gemini-4-argon
        name: Gemini 4 Argon
        contextWindow: 1048576
        maxTokens: 32768
        efforts: { low: LOW, medium: MEDIUM, high: HIGH }
      - id: gemini-3.8-flash
        maxTokens: 65536
        efforts: { minimal: MINIMAL, low: LOW, medium: MEDIUM, high: HIGH }
      - id: gemini-2.5-flash
        maxTokens: 65536
        efforts: { low: 1024, high: 24576 }   # 数字 = thinkingBudget
```

注意：`models` 是整个列表替换，不是追加，想保留默认模型就要一起写上。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `provider` | `aistudio` | 路由 id（模型选择器、`agent-default-model` 里用） |
| `displayName` | `AI Studio` | 选择器里显示的提供方名 |
| `apiKeyEnv` | `GEMINI_API_KEY` | key 的凭据引用名 |
| `baseURL` | `https://generativelanguage.googleapis.com/v1beta` | API 根地址 |
| `models` | 见上表 | 模型列表；`efforts` 的值为 thinkingLevel（MINIMAL/LOW/MEDIUM/HIGH）或 token 预算（数字） |
| `defaultEffort` | `high` | 请求和模型都没指定时的推理强度 |
| `includeThoughts` | `true` | 请求思考摘要，在 UI 里显示为 reasoning |
| `streamIdleTimeoutMs` | `300000` | 流超过这么久没有数据就判定 TIMEOUT |

默认模型示例：

```yaml
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: aistudio
    model: gemini-4-argon
    reasoningEffort: high
```

## 行为说明

- **finishReason 全覆盖**：`STOP` → 正常结束（有工具调用则为 tool-calls，没有可见输出则为可重试的 `EMPTY_RESPONSE`）；`MAX_TOKENS` → max-tokens；`SAFETY`/`RECITATION`/`PROHIBITED_CONTENT` 等 → `CONTENT_FILTERED`（不重试）；`MALFORMED_FUNCTION_CALL`、`MALFORMED_RESPONSE`、`OTHER` 以及**任何未知的新 reason** → `MALFORMED_RESPONSE`（自动重试）。不会再出现 "Unhandled stop reason"。
- **重试策略**：`EMPTY_RESPONSE`、`RATE_LIMIT`、`SERVER`、`TIMEOUT`、`TRANSPORT`、`MALFORMED_RESPONSE` 最多重试 4 次（1–20 s 退避）；429 会遵守 Google 返回的 `retryDelay`；按天的配额耗尽归为 `QUOTA`，不重试。
- **thought signature**：签名按原来的 part 原样回传，只回传给同一路由、同一模型；挂在空文本 part 上的签名也会回传。
- **跨模型切换**：Gemini 3 及以上要求当前 turn 里的 functionCall 带签名，否则返回 400（"Function call is missing a thought_signature"）。从别的模型或 provider 切过来时，当前 turn 中缺签名的工具调用会填 Google 文档给出的绕过值 `skip_thought_signature_validator`（已实测：不填会 400，填了正常）。
- **工具**：使用 `parametersJsonSchema`（完整 JSON Schema），`const` 改写成单值 `enum`；Gemini 3 及以上的 functionCall/functionResponse 带 `id`；并行工具结果合并在同一个 user turn。
- **图片**：用户消息和工具结果里的图片以 inlineData 发送（Gemini 3 及以上放进 functionResponse.parts）；单张图片目标不超过 4 MiB，整个请求的图片不超过 14 MiB（base64）、100 张，超出时返回 `IMAGE_OFFLOAD_REQUIRED`，由 dsh 卸载最早的图片后重试。
- **错误分类**：无效 key → `INVALID_CREDENTIAL`；超出上下文 → `CONTEXT_WINDOW_EXCEEDED`（触发压缩）；401/403 → `AUTH`；404 → `UNKNOWN_MODEL`；5xx → `SERVER`。
- **代理**：用的是普通 `fetch`，dsh 启动器会按 `HTTPS_PROXY` 自动走代理；不设代理时直连。

## 已知限制

- 不声明 `systemPromptUpdate`/`toolUpdate`：每次请求都带完整工具列表；system prompt 中途变化时，后续的 system 消息以 `<system-reminder>` 形式放进 user turn。
- 不支持 Files API、code execution、Google Search grounding 等 Gemini 内置工具。
- 模型目录来自配置，不会自动从 `/v1beta/models` 发现。

## 开发

```sh
npm run link-dev   # 把全局 dsh 的 @deepseek-ai/* peer 包软链接进 node_modules
npm test           # 单元测试（不联网）
npm run smoke      # 真实 API 冒烟测试：text / tools / bypass / image，key 从凭据库读取
```
