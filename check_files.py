from pathlib import Path

files_to_check = [
    "src/agents/coder.py",
    "src/agents/playbooks/coder.md",
    "src/agents/base_agent.py",
    "src/agents/analyst.py",
    "src/utils/llm_client.py",
    "src/utils/token_ledger.py",
    "src/services/pipeline_flow.py",
    "src/services/step_adapters.py",
    "config/pipelines.json",
    "main.py",
]

print("=" * 60)
print("FILE EXISTENCE CHECK")
print("=" * 60)
for f in files_to_check:
    path = Path(f)
    status = "✅ EXISTS" if path.exists() else "❌ MISSING"
    print(f"{status}: {f}")
print("=" * 60)
