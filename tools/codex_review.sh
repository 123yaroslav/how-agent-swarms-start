#!/bin/zsh
# Codex review gate for the report. Usage: tools/codex_review.sh <tag> [extra focus text]
# Renders the current PDF text, asks Codex (read-only, no network, no execution) for findings
# against the numbers, plans and code, and saves docs/reviews/codex_<tag>_review.md.
# Rule: a MAJOR/HIGH finding blocks the commit until fixed and re-reviewed.
set -e
cd "$(dirname "$0")/.."
TAG=$1; FOCUS=${2:-}
pdftotext -layout report/Apart_report.pdf docs/reviews/rendered_$TAG.txt
cat > docs/reviews/codex_${TAG}_prompt.txt <<PROMPT
READ-ONLY review. You may run read-only shell commands (cat, sed, grep, ls, head, tail) to read files; do not modify files, do not run project code, no network. <= 450 words, numbered findings with severity (BLOCKER/MAJOR/MINOR) and a short quote to locate each.
Review the current report text (rendered: docs/reviews/rendered_$TAG.txt; source: report/content.md with {{placeholders}} filled from report/numbers.json).
Check, in this order:
1. Every number and verdict in the body against report/numbers.json and the pre-registered plans (docs/analysis_plan.md, docs/confirmatory_plan.md, docs/agrawal_live_plan.md, docs/fresh_plan.md). Post-hoc results must be labelled as such.
2. Formulas in Section 3 and Appendix C.9 against the code: src/collusion/detect/net_influence.py, src/collusion/detect/crt_llm.py, src/collusion/detect/gro.py, src/collusion/detect/departure.py.
3. Caveats that an earlier version carried and this one lost (exchangeability not established, routing unpinned, fixed-horizon calibration empirical, sequential test built after the freeze, judge non-determinism, k=2 pre-registered, pilots descriptive, direct effect only).
4. Overclaims in the abstract, headings, captions and conclusion; any sentence that promises more than its table supports.
5. Internal consistency: figure/table/section references, numbering of propositions and appendices, citation numbers against the reference list order.
$FOCUS
PROMPT
codex exec -s read-only -m gpt-6-luna -c model_reasoning_effort=high "$(cat docs/reviews/codex_${TAG}_prompt.txt)" < /dev/null > docs/reviews/codex_${TAG}_raw.txt 2>&1 || true
n=$(grep -n "^codex$" docs/reviews/codex_${TAG}_raw.txt | tail -1 | cut -d: -f1)
tail -n +$((n+1)) docs/reviews/codex_${TAG}_raw.txt | grep -v "^tokens used" | awk 'NF' > docs/reviews/codex_${TAG}_review.md
echo "REVIEW_DONE $TAG: $(grep -c -E 'BLOCKER|MAJOR|HIGH' docs/reviews/codex_${TAG}_review.md) major findings"
