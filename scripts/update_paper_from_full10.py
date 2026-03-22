#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path('/teamspace/studios/this_studio')
SUMMARY_DIR = ROOT / 'text_to_sign' / 'conditional_loss_audit_20260321'
RESP_PATH = ROOT / 'text2sign_paper' / 'RESPONSE_TO_REVIEWERS.md'

summaries = sorted(SUMMARY_DIR.glob('full10_posteval_summary_*.md'))
if not summaries:
    raise SystemExit('No full10 summary found yet.')

latest = summaries[-1]
text = latest.read_text()

# Pull key values from the summary markdown lines.
def extract(label: str) -> str:
    m = re.search(rf"{re.escape(label)}: ([0-9.+-]+)", text)
    return m.group(1) if m else 'N/A'

short_top1 = extract('top1 accuracy mean')
short_rank = extract('mean correct rank')
short_none = extract('none-minus-normal mean')
short_rand = extract('random-minus-normal mean')

# second block values: use findall and take second occurrence when available
all_top1 = re.findall(r"top1 accuracy mean: ([0-9.+-]+)", text)
all_rank = re.findall(r"mean correct rank: ([0-9.+-]+)", text)
all_none = re.findall(r"none-minus-normal mean: ([0-9.+-]+)", text)
all_rand = re.findall(r"random-minus-normal mean: ([0-9.+-]+)", text)
large_top1 = all_top1[1] if len(all_top1) > 1 else 'N/A'
large_rank = all_rank[1] if len(all_rank) > 1 else 'N/A'
large_none = all_none[1] if len(all_none) > 1 else 'N/A'
large_rand = all_rand[1] if len(all_rand) > 1 else 'N/A'

resp = RESP_PATH.read_text()
anchor = '## Additional manuscript-level actions completed\n'
if anchor not in resp:
    raise SystemExit('Could not locate insertion anchor in RESPONSE_TO_REVIEWERS.md')

insertion = f"""
## Full 10-epoch post-revision addendum (new)

After this revision cycle, we completed an extended 10-epoch follow-up training run under the updated text-conditioning architecture path and reran the held-out conditional-loss audits.

- Short audit (high-t, short prompts): top-1={short_top1}, mean-rank={short_rank}, none-minus-normal={short_none}, random-minus-normal={short_rand}
- Larger audit (high-t, short prompts): top-1={large_top1}, mean-rank={large_rank}, none-minus-normal={large_none}, random-minus-normal={large_rand}

These full-run follow-up values are now used only to refine the paper’s conditioning discussion conservatively (evidence of measurable text effect remains stronger for no-text vs normal than for prompt-identity separation), not to claim solved semantic-faithful sign generation.

"""

if '## Full 10-epoch post-revision addendum (new)' in resp:
    # Replace existing addendum block conservatively.
    resp = re.sub(
        r"## Full 10-epoch post-revision addendum \(new\)[\s\S]*?(?=\n## Additional manuscript-level actions completed)",
        insertion.strip() + "\n\n",
        resp,
    )
else:
    resp = resp.replace(anchor, insertion + anchor)

RESP_PATH.write_text(resp)
print(f'Updated {RESP_PATH} from {latest}')
