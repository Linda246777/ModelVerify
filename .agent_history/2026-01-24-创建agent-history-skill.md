# Agent History

## 2026-01-24 - 创建 Claude Code Agent History Skill

### 模块变更总结

- 删除项目中的 `.agent/` 目录（skill 应放在全局位置）
- 在 `~/.claude/skills/agent-history/` 创建全局通用的会话历史记录 skill
- 更新 `CLAUDE.md` 添加 Agent Tools 说明

### 对话过程

1. 用户要求创建可在 Claude Code 中通用的 skill，而非项目特定脚本
2. 我搜索研究 Claude Code 的 skill 创建机制
3. 确认 skill 应存储在 `~/.claude/skills/<skill-name>/SKILL.md`
4. 了解 skill 的 YAML frontmatter 格式（name、description、user-invocable、argument-hint 等）
5. 创建 `~/.claude/skills/agent-history/SKILL.md` 文件
6. 删除项目中的 `.agent/` 目录
7. 更新 `CLAUDE.md` 添加 Agent Tools 说明
8. 用户要求记录对话历史
9. 我创建了 `.agent_history/2026-01-24-创建agent-history-skill.md`

### 关键决策

- Skill 存储在全局目录 `~/.claude/skills/`，便于跨项目使用
- 使用 `user-invocable: true` 使 skill 可通过 `/` 菜单调用
- 使用 `argument-hint` 提供参数提示

### 技术细节

- Skill 格式：YAML frontmatter + Markdown 说明
- 调用方式：`/agent-history "会话摘要"`
- 自动生成文件名：`.agent_history/YYYY-MM-DD-标题.md`

### 下一步提示

- 可以在 `/` 菜单中看到新的 skill
- 后续会话结束时可使用此 skill 记录历史
