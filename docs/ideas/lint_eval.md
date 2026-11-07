# SwarmCheck lint on real framework prompts: held-out evaluation (EXPLORATORY)

Generated from results/realworld_prompts/lint_eval_v1.json and lint_eval.json by the command in this file's commit; numbers are not edited by hand. Labels: hand labels committed in 78c2c1b before any linter run. Split by project: md5(project) even = dev (13 projects, 26 prompts), odd = test (15 projects, 25 prompts). Regex v2 was tuned on dev only and committed (3a86a86) before it was run on test. The LLM rubric (src/swarm/swarmcheck/lint_llm.py) was written from the label definitions before any run and not tuned. The test split has now been used once for each method and is spent; later changes need new prompts.

| split | method | contact found (recall, Wilson 95%) | contact precision | missing rule found (recall) | missing rule precision |
|---|---|---|---|---|---|
| test | regex v1 (written for our harness prompts) | 0/4 [0.00, 0.49] | n/a | 0/6 [0.00, 0.39] | n/a |
| test | regex v2 (tuned on dev) | 0/4 [0.00, 0.49] | n/a | 1/6 [0.03, 0.56] | 1/1 |
| test | LLM rubric, deepseek-v4-flash | 3/4 [0.30, 0.95] | 3/4 | 0/6 [0.00, 0.39] | n/a |
| test | LLM rubric, glm-5.2 | 2/4 [0.15, 0.85] | 2/3 | 1/6 [0.03, 0.56] | 1/1 |
| dev | regex v1 (written for our harness prompts) | 0/7 [0.00, 0.35] | n/a | 2/8 [0.07, 0.59] | 2/2 |
| dev | regex v2 (tuned on dev) | 6/7 [0.49, 0.97] | 6/6 | 7/8 [0.53, 0.98] | 7/7 |
| dev | LLM rubric, deepseek-v4-flash | 7/7 [0.65, 1.00] | 7/8 | 5/8 [0.31, 0.86] | 5/5 |
| dev | LLM rubric, glm-5.2 | 5/7 [0.36, 0.92] | 5/6 | 5/8 [0.31, 0.86] | 5/5 |
| all | regex v1 (written for our harness prompts) | 0/11 [0.00, 0.26] | n/a | 2/14 [0.04, 0.40] | 2/2 |
| all | regex v2 (tuned on dev) | 6/11 [0.28, 0.79] | 6/6 | 8/14 [0.33, 0.79] | 8/8 |
| all | LLM rubric, deepseek-v4-flash | 10/11 [0.62, 0.98] | 10/12 | 5/14 [0.16, 0.61] | 5/5 |
| all | LLM rubric, glm-5.2 | 7/11 [0.35, 0.85] | 7/9 | 6/14 [0.21, 0.67] | 6/6 |

Reading. Hand-written patterns did not transfer between projects: tuned on half of them they found 6 of 7 contact prompts there and none of the 4 in the other half. The LLM rubric with deepseek-v4-flash found 3 of 4 held-out contact prompts with 1 false alarm. Neither method finds missing boundary rules reliably on held-out prompts, because both miss what counts as shared storage in unfamiliar wording. The test split is small (4 contact prompts, 6 missing-rule prompts), so these are weak evidence. The tool therefore ships with the LLM rubric as the default for contact, and with the regex rules as a fast pre-filter that can only add findings.
