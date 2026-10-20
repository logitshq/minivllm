# Batched GDN graph composition — prepared only

Both switches default OFF. `install(engine, gdn_prework=True, route_graph=True)`
uses E033's unchanged fresh-instance ownership/rollback/graph teardown functions
in a private namespace and replaces only the GDN adapter with E046's B1–8 source.
No original module is modified. The native scheduler, weights, five batched
projections and graph implementation stay as pinned in the existing composition.

E046 qualified local real-activation B2/B8 bytes; E048 screens local eager cost.
Neither qualifies this new graph composition or its other batch sizes. Before
HTTP timing, compare full logits/state to native under the same actual schedule,
verify real candidate tracing and continued state, and retain all failures.
The proposed first canary is the observed E047 warm prefix through step49,
after the original graph-only prefix is validated independently. No device run
or source-linked speed claim exists for this module yet.

The CPU tests reuse the original fake-model lifecycle suite to check all four
switch combinations, install-before-trace, rollback, graph clearing, restore
order and private module globals. They do not execute tensor arithmetic.
