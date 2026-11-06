# Feasibility check, topic 2 (takedown as a natural experiment)

Checked 2026-11-06 on `data/raw/collusion_wiki/x/`. Every number below is printed by one of the `experiments/feas_t2_*.py` scripts (run with `uv run python -I`). Hourly, daily and pool tables are in `results/feas_t2/*.csv`. No LLM calls.

**Guardrail kept.** No outcome of a deletion was computed. That means no activity, recreation, renaming or encoding of treated vs untreated units after deletion, and no event-study curves. The only exceptions are the ones the brief allows, the investigators' 68 labelled edges (step 3) and the aggregate awareness counts (step 5). Section 7 lists everything that was seen.

## Verdict: CHANGE (neither go as designed nor no-go)

| Gate in `topic2_takedown.md` | Result | Pass? |
|---|---|---|
| >= 100 active treated units in the window | 207 deleted pages had a save in the 48 h before deletion (06-16..06-21). 199 of them have >= 24 h before the global stop | Yes |
| Recreation search finds >= half of the 67 labelled cases | Content matcher 10/60 (containment >= 0.5). The labels are "first later edit of the same page name", not content copies (section 3) | No, but the gate is mis-specified |
| Enough similar control units | Never-deleted dse pages: 10 (8 active). The only workable controls are not-yet-treated pages (later deleted). The pool is 1,271-1,694 active pages per hour on 06-19 | Only staggered |

## 1. Treatment timing and units

**Daily, 06-01..07-14** (days with no deletions are omitted. "Pre-body" means the page has a stored revision before the deletion. "Active" means it has a save in the 48 h before. "Labels" counts distinct non-blank labels with a revision on the page before the deletion.)

| Day | Delete events | Distinct pages | With pre-body | Active 48 h | Labels |
|---|---|---|---|---|---|
| 06-04 | 2 | 2 | 1 | 0 | 1 |
| 06-18 | 25 | 19 | 18 | 18 | 97 |
| 06-19 | 317 | 309 | 232 | 154 | 475 |
| 06-20 | 78 | 72 | 58 | 32 | 171 |
| 06-21 | 11 | 11 | 11 | 10 | 127 |
| 06-22 | 11 | 11 | 10 | 10 | 32 |
| 06-23 | 602 | 602 | 456 | 70 | 437 |
| 06-24 | 267 | 267 | 241 | 0 | 606 |
| 06-25 | 179 | 179 | 154 | 0 | 208 |
| 06-26 | 382 | 382 | 310 | 0 | 539 |
| 06-28 | 146 | 146 | 93 | 0 | 152 |
| 06-29 | 88 | 88 | 74 | 0 | 110 |
| 06-30 | 440 | 440 | 311 | 0 | 422 |
| 07-01 | 248 | 248 | 234 | 0 | 308 |
| 07-02 | 88 | 88 | 88 | 5 | 46 |
| 07-03 | 71 | 71 | 71 | 0 | 42 |
| 07-04 | 59 | 59 | 59 | 0 | 35 |
| 07-05 | 239 | 239 | 211 | 0 | 257 |
| 07-06 | 79 | 79 | 64 | 0 | 84 |
| 07-07 | 522 | 522 | 383 | 0 | 675 |
| 07-08 | 230 | 230 | 155 | 0 | 220 |
| 07-09 | 154 | 153 | 88 | 0 | 132 |
| 07-11 | 86 | 86 | 49 | 0 | 79 |
| 07-12 | 232 | 232 | 153 | 0 | 148 |
| 07-13 | 512 | 511 | 333 | 0 | 333 |
| 07-14 | 149 | 149 | 88 | 0 | 85 |

**Hourly, 06-16..06-23 (UTC), hours with deletions.** There were none on 06-16 or 06-17.

| Hour | Events | Pages | Pre-body | Active 48 h | Labels |
|---|---|---|---|---|---|
| 06-18T18 | 25 | 19 | 18 | 18 | 97 |
| 06-19T13 | 59 | 59 | 38 | 25 | 97 |
| 06-19T14 | 24 | 24 | 15 | 4 | 31 |
| 06-19T15 | 41 | 41 | 36 | 32 | 78 |
| 06-19T16 | 38 | 38 | 37 | 31 | 124 |
| 06-19T22 | 8 | 8 | 7 | 7 | 45 |
| 06-19T23 | 147 | 144 | 104 | 60 | 175 |
| 06-20T00 | 75 | 70 | 56 | 31 | 114 |
| 06-20T09 | 1 | 1 | 1 | 0 | 18 |
| 06-20T19 | 2 | 2 | 2 | 2 | 61 |
| 06-21T11 | 6 | 6 | 6 | 6 | 90 |
| 06-21T12 | 2 | 2 | 2 | 1 | 23 |
| 06-21T21 | 3 | 3 | 3 | 3 | 25 |
| 06-22T19 | 11 | 11 | 10 | 10 | 32 |
| 06-23T11..23 | 602 | 602 | 456 | 70 | (see CSV) |

**Units**

| Quantity | Value |
|---|---|
| Delete events / distinct pages | 5,217 / 5,144 (all dse, all `[Admin1]`) |
| Deleted pages absent from the export (no stored revision at all) | 1,246 (1,248 events) |
| Deleted pages with a stored pre-deletion revision | 3,895 |
| First deletions 06-16..06-21 / active in prior 48 h | 403 / 207 |
| Active treated pages with >= 24 h before the global stop | 199 |
| Deletion batches (consecutive hours merged), active pages | 06-18T18 18, 06-19T13-16 91, 06-19T22-06-20T00 90, 06-21T11-12 5, 06-21T21 3 |
| Never-deleted pages | 681 (probier 601, fractal 68, dse 10, dorfwiki 2) |
| Never-deleted pages with a save in 06-16..06-21 | 456 (probier 425, fractal 23, dse 8) |
| Labels touched by a deletion / never touched | 2,886 / 216 of 3,102 non-blank labels |
| Labels first touched 06-17..06-21 | 695 |

**Not-yet-treated pool per hour** (pages with a save in the prior 48 h and not deleted by the end of the hour, min-max over the day's hours. Full table in `pool_hourly_0617_0621.csv`.)

| Day | Later-deleted dse (main pool) | Never-deleted dse | Never-deleted probier/fractal | Not-yet-treated active labels |
|---|---|---|---|---|
| 06-17 | 742-1,135 | 3-4 | 17-28 | 716-1,043 |
| 06-18 | 1,084-1,795 | 4-7 | 28-393 | 1,020-1,344 |
| 06-19 | 1,271-1,694 | 6-7 | 387-397 | 680-1,240 |
| 06-20 | 244-1,368 | 4-6 | 40-399 | 201-787 |
| 06-21 | 228-303 | 2-4 | 31-40 | 190-291 |

**Swarm stop.** Saves run at 0-59 per hour through 06-21. On 06-22 there are bursts of 411 (02h) and 463 (08h), and the last save before the gap is at **06-22 09:20:04Z**, which is exactly the manifest's recreation cutoff (`cutoff_unix_ts` 1782120004). After that there is 1 save on 06-22 (19h), 1 on 06-23, 1 on 06-24, 7 on 07-01 and 14 on 07-02. So 4,784 of the 5,217 deletions (06-22 onward) hit a swarm that had already stopped, and they carry no usable post-period.

## 2. How a deletion maps to pages and revisions

| Question | Answer |
|---|---|
| Delete record | `events.jsonl`, `event_type=delete`, `event_id=delete:dse:rclog:<n>`, `page_key` (`dse~Name`), `time` (1 s, 5,216 reqlog-corroborated, 1 rclog-only), `actor_label=[Admin1]`, `change_summary="Seite gelöscht."`, `page_held` |
| Link to revisions | Implicit only. `revision_ref` and `related_event_id` are null on deletes. The removed content is the stored revisions with the same `page_key` and `time` before the delete (and after any earlier delete of that page) |
| Deleted body available? | Yes, for 3,966 of 5,217 events: the full `body` of the last stored pre-deletion revision. It is missing for 1,248 events (page not in export) and for 3 events (page held, but its only revisions come after the delete). Caveat: edits under 64 chars, non-agent edits and most moderator saves are not stored (section 4), so the last stored body may differ from what was on the page at deletion time |
| Live-state fields | `pages.deleted_live` is False for all 4,579 pages, so it does not mark deletion. `pages.n_deletions` agrees with the event counts for 4,579/4,579 pages |
| History across deletion | The RCS chain continues. For 61 of 61 evaluable labelled recreations, `seq` keeps counting and `diff_base` is the pre-deletion revision, so the hunks of a post-deletion save are a diff against the deleted text |
| Multiple deletions | Yes. Pages deleted 1/2/3/5/8 times: 5,089/44/9/1/1 |
| `first_recreation_of` edges | 67 events carry it (63 `save`, 4 `revert`) and give 68 relations, because one save links to 2 deletes 28 s apart. `related_event_id` is a list on saves and a string on reverts. The same 63 revisions carry the relation in `revisions.jsonl` |
| What an edge links | Delete event to the **first later successful mutation of the same page name** (68/68 same `page_key`, 68/68 later). The rule is derived from the rclog ("first-later-edit rule"). It links the same page/label/IP save within 2 s when held, otherwise the first later stored non-admin save (58 exact, 6 fallback, 4 reverts without a stored revision). There are 50 distinct pages. Every edge page was deleted >= 2 times. The deletes run 06-04 to 06-20T19:40 and the recreations 06-08 to 06-22T02:53. The revert actors are agent labels, not the moderator |

## 3. Positive control: content matcher

Matcher (`feas_t2_matcher.py`): lowercase, strip `-- Name` lines, mask digits, word 3-shingles, Jaccard J and containment C (share of the deleted body's shingles found in the candidate). Query: the last stored non-admin, non-empty pre-deletion body. Candidates: all revisions after the deletion on any page.

| Test | n | J >= 0.5 | C >= 0.5 | C >= 0.3 |
|---|---|---|---|---|
| Labelled edges, target scored at threshold (recall) | 60 evaluable of 68 (6 no pre-body, 2 reverts with no stored target) | 8 | 10 | 14 |
| Labelled edges, target in top 1 / 5 / 10 of all later revisions by J | 60 | 10 / 14 / 16 | | |
| Random cross-page pairs, candidate written **before** the anchor's deletion (false matches) | 200 | 0 | 1 | 7 |
| Hard pairs: same `page_family`, candidate among the 3,000 saves before the deletion | 200 | 2 | 10 | 19 |

The negative rows come from one seeded draw. An earlier draw (same seed, with admin/empty anchors kept) gave 0/0/5 and 1/6/27, so read hard-pair false matches at C >= 0.5 as about 3-5%.

Why recall is low (`feas_t2_matcher_diag.py`, no digit masking, n = 61). The labelled targets are mostly new text on a reused page name. Median char-5-gram Jaccard is 0.10. Lines copied from the deleted body have a median of 0. Only 2 targets are byte-identical to the deleted body, only 9 share its label, and even the best of three representations (char 5-gram Jaccard, containment, copied lines) reaches >= 0.5 for only 18/61. So the 68 edges validate a **page-name-reuse** outcome (which they define), not a content-recreation outcome. They cannot test cross-page detection, because all of them are same-page.

Synthetic positive control (`feas_t2_matcher_synth.py --mask-digits`, 200 random pre-deletion bodies with >= 20 words, no post-deletion data used):

| Perturbation | Recall J >= 0.5 | Recall C >= 0.5 |
|---|---|---|
| Exact copy, new signature | 200/200 | 200/200 |
| Drop 30% of lines | 168/200 | 168/200 |
| Random digits | 200/200 | 200/200 |
| Append an unrelated page | 90/200 | 200/200 |
| Keep half + unrelated page | 42/200 | 131/200 |
| Drop 30% + digits + append | 68/200 | 164/200 |

Containment >= 0.5 with digit masking works as an instrument (164-200/200 on plausible edits, 1/200 random and 10/200 hard false matches). But only a synthetic or hand-labelled set can validate it on cross-page copies.

## 4. Attribution

| Fact | Value |
|---|---|
| Inclusion rule | No per-revision classifier is documented in the export. The population is `farm.sqlite` revisions with `write_date >= 2026-05-01`. It keeps 3 labels allowlisted as pre-existing human handles, tokenises human hosts, and replaces pre-cut lines of 17 legacy pages with a placeholder. `records.jsonl` authorship is `not_independently_authenticated` for all 13,703 |
| Revisions with a label | 13,692 / 14,591 (93.8%). 899 blank |
| With a valid ip16 | 14,591 / 14,591 |
| With a `-- Name` signature line in the added text | 3,638 (24.9%). The signature equals the save label in 2,512 |
| Human-handle labels (`is_human_handle`) | 3: `[Admin1]` 26 revisions, `[Admin2]` 4, `[Admin2]302` 1 |
| Moderator revisions in the export | 26 `[Admin1]` saves on 6 pages, 06-02..06-24, nearly all "revert to revision 1" of legacy pages. Two have empty bodies. The `[Admin2]` texts look like agent test posts (a handle collision or impersonation, unresolved) |
| **Unstored moderator saves** | `labels.jsonl` counts **3,887 `[Admin1]` save requests on 3,789 pages** from 2 hosts (the manifest gives 3,889 admin form edits), against 26 stored. Their times and content are not in the export |
| Label quality | 1,332 labels have 1 revision. The top label has 899 revisions from 741 IPs. `AgentRelent` has 317 revisions from 308 IPs on 4 pages. Labels are not agents |

## 5. Exposure, interference, awareness

| Check | Finding |
|---|---|
| Agent page reads | Not observable. The export has no per-agent fetch logs. `site_figures/visits.js` gives daily agent edits and daily OpenAI-employee requests only. The shortener log (499 links, 05-26..06-21, time and ip16) records link creation, not reads (`clicks` is a total at read date). Treatment is intent-to-treat: we cannot know whether an agent saw that its page was gone |
| Caching | Agents also used cache/relay pages (`page_family` source-cache 1,231 pages), so a deleted page's content may stay reachable through copies made before deletion. This is exactly what the hard-pair false matches (10/200 at C >= 0.5) pick up, so a placebo is required |
| Awareness words (revision added text, `[Admin1]` excluded, all units pooled) | Revisions with any term per day: <=06-15 2, 06-16 2, 06-17 4, 06-18 1, 06-19 6, 06-20 9, 06-21 8, 06-22 0, >=06-23 0. By term, all days: delete* 27, removed 2, admin 4 (all on 06-21), moderator 0, gelöscht/lösch* 0, wiped 0. Daily saves: 2,601 / 1,294 / 6,536 / 505 / 657 / 655 / 1,070 for 06-16..06-22 |
| Awareness words, other venues (`records.jsonl`, 9,618 dated) | Records with any term: 06-17 1, 06-18 1, 06-19 5, 06-20 7, 06-21 1, other days 0. moderator 0, gelöscht 0 |

Awareness in text is near zero, so a pre-registered "mentions of the moderator" outcome would be a floor effect.

## 6. Minimum detectable effect (counts only)

The page outcome is binary, "saved again within 24 h". Two-sided alpha 0.05, power 0.8, no clustering (`feas_t2_mde.py`). The base rate was measured only before the first deletion batch, for planning: the share of active pages saved again within 24 h is 0.417 at 06-16T12, 0.080 at 06-17T00 and 0.027 at 06-17T12. Pages are mostly one-shot and the base rate drifts fast.

| p0 | nT = 199, nC = 1,000 | nT = 100, nC = 1,000 |
|---|---|---|
| 0.03 | 3.7 pts | 5.0 pts |
| 0.05 | 4.7 pts | 6.4 pts |
| 0.08 | 5.9 pts | 8.0 pts |
| 0.20 | 8.7 pts | 11.8 pts |

At a realistic p0 of 0.03-0.08 the MDE is about the same size as p0. Only near-total suppression or a doubling is detectable. Treatment also arrives in 3 batches with >= 10 pages (18, 91, 90), so cluster-level inference has about 3 effective shocks. Use randomization inference over placebo batch times instead of unit-level standard errors.

## 7. Recommendation

**Change, then pre-register.** Concretely:

1. **Primary unit and window.** Use dse pages with first deletion 06-18T18..06-21T21 that were active in the prior 48 h (n = 207, 199 with >= 24 h of post-window). Compare with not-yet-treated active dse pages (later deleted) in the same hour. Drop never-treated controls: 8 dse pages, and probier/fractal have no moderator. Censor everything at 06-22 09:20:04Z.
2. **Redefine the recreation outcome and its positive control.** Report two outcomes. (a) Page-name reuse, as the investigators define it. The 68 edges are its definition, so they are not a validation. (b) Content recreation, measured by digit-masked word-3-shingle containment >= 0.5 against the deleted body on any page. Validate (b) on the synthetic set above, plus a small hand-labelled set (for example 30 matcher hits and 30 near-misses, labelled blind to treatment). Its false-match baseline is the pre-deletion hard pairs (10/200), so a placebo with fake deletion times is required.
3. **Inference.** Use a staggered design with not-yet-treated controls and randomization inference over the 5 batch times. Pre-register the MDE table above. Make the null result ("no detectable change, MDE about p0") an acceptable headline.
4. **Drop or demote.** Make the "mentions of the moderator" adaptation outcome descriptive only (it is at the floor). Make the label-level displacement outcome secondary (1,332 single-revision labels, labels shared across hundreds of IPs).
5. **Open validity risk to resolve or disclose.** 3,887 moderator save requests on 3,789 pages are counted but not timed or stored. If they edited or blanked pages before the delete events, the treatment onset is earlier than the delete time for many pages. Ask the collusion.wiki authors, or check their report for how ProWiki deletion works. Until then, add a robustness window that starts treatment 1 h before the delete.

**Disclosure for `docs/prereg.md` (what was seen).** Daily whole-swarm saves (already in `data_recon.md`). Hourly saves for 06-21..06-22 (stop check). Aggregate daily counts of deletion/moderator words (section 5). The pre-period persistence base rate (section 6). Similarity scores of the 68 labelled edges (section 3). Also, by accident, the save/delete event sequence of two labelled-edge pages (deleted 8 and 5 times); the print was then removed from `feas_t2_edges.py`. Nothing was seen on post-deletion activity, recreation, renaming or encoding of treated vs untreated units.
