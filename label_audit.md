# Label Audit

Scope:
- Manual review of disagreement-heavy diagnostics cases
- Classification based on query semantics first, not current expected labels

| query | old | corrected | confidence |
| --- | --- | --- | --- |
| kubernetes crashloopbackoff memory limit fix | NARROW | NARROW | high |
| what api login issues are we seeing | AMBIGUOUS | AMBIGUOUS | medium |
| deployment failures and rollback incidents | AMBIGUOUS | AMBIGUOUS | medium |
| build and deploy pipeline problems | AMBIGUOUS | AMBIGUOUS | medium |
| database and orders query issues | AMBIGUOUS | AMBIGUOUS | high |
| team planning and handoff process issues | AMBIGUOUS | AMBIGUOUS | medium |
| pipeline timeout versus docker build bloat | AMBIGUOUS | AMBIGUOUS | high |
| give me all recent operational updates across teams | BROAD | BROAD | high |
| summarize all infrastructure and devops updates | BROAD | BROAD | high |
| what happened across security topics lately | BROAD | BROAD | medium |
| give me engineering updates across backend frontend and mobile | BROAD | BROAD | high |
| summarize all auth and access related updates | BROAD | BROAD | medium |
| what were the main reliability issues across the system | BROAD | BROAD | high |
| give me all deployment and release related updates | BROAD | BROAD | high |
| weather forecast for mumbai this weekend | REJECT | REJECT | high |

Takeaways:
- The current benchmark labels do not look obviously broken in the selected disagreements.
- The biggest visible issue is not label corruption; it is classifier collapse of valid `AMBIGUOUS` and `BROAD` queries into `REJECT` or `NARROW`.
- The boundary between `AMBIGUOUS` and `BROAD` is still the least crisp. Several of those rows are only medium-confidence even after manual review.

One-sentence working definitions:
- `NARROW`: one concrete issue or thread is clearly implied by the query.
- `AMBIGUOUS`: the query plausibly points to a small number of distinct specific threads.
- `BROAD`: the query explicitly asks for a summary across multiple topics or threads.
- `REJECT`: the query is outside the Slack archive domain.
