# https://platform.claude.com/docs/ko/agents-and-tools/agent-skills/quickstart

import anthropic

client = anthropic.Anthropic()

# List Anthropic-managed Skills
skills = client.beta.skills.list(
  source="anthropic",
  betas=["skills-2025-10-02"]
)

for skill in skills.data:
  print(f"{skill.id}: {skill.display_title}")
