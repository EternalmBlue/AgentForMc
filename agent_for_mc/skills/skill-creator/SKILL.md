---
name: skill-creator
description: Use only during skill authoring flows to help a server owner turn requirements into a concise AgentForMc SKILL.md.
usage: authoring
---

# Skill Creator

## Workflow

1. Understand the server owner's operational goal and the exact situations where the skill should trigger.
2. Ask only for missing details that materially change the skill: trigger conditions, workflow, evidence sources, output shape, and safety constraints.
3. Generate one focused Markdown skill with YAML frontmatter containing `name` and `description`.
4. Keep the body concise. Prefer workflow rules and output rules over broad background explanation.
5. Do not create scripts, executable resources, secret handling instructions, or rules that bypass system safety.

## Output Rules

- Use a kebab-case skill name.
- Make the description specific enough for routing.
- Include trigger conditions, workflow, evidence preferences, and response constraints.
- If the user's request is too broad, ask clarification questions before drafting.
