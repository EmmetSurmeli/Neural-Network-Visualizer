"""Enforce repository-neutral whitespace and parse committed deployment config."""
import json
import subprocess
from pathlib import Path

errors = []
for name in subprocess.check_output(['git', 'ls-files'], text=True).splitlines():
    path = Path(name)
    if not path.exists() or path.suffix not in {'.py', '.ts', '.tsx', '.css', '.md', '.json', '.yml', '.yaml'}:
        continue
    text = path.read_text()
    # Existing generated datasets intentionally omit the terminal newline.
    generated = path.suffix == '.json' and (name.startswith('backend/models/') or name == 'examples/digit-7.json')
    if (text and not generated and not text.endswith('\n')) or any(line.rstrip() != line for line in text.splitlines()):
        errors.append(f'{name}: use a final newline and remove trailing whitespace')
    if path.suffix == '.json':
        json.loads(text)
if errors:
    raise SystemExit('\n'.join(errors))
print('Whitespace and JSON checks passed.')
