import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "reviews_processed.json"

with open(PROCESSED_FILE) as f:
    data = json.load(f)

issues = []
for r in data:
    cats = r.get('category_tags', [])
    descs = r.get('descriptor_tags', [])
    for tags in [cats, descs]:
        for t in tags:
            if t.startswith('!'):
                base = t[1:]
                if base in tags:
                    issues.append((r.get('review_id', ''), base, tags))

if issues:
    print(f'Found {len(issues)} issues:')
    for rid, tag, tags in issues[:5]:
        print(f'  {rid[:50]}: {tag} in {tags}')
else:
    print('No duplicate tag/!tag issues found!')
print(f'Total processed: {len(data)}')