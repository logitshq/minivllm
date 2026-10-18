# Cold cycle after independently observed service recovery

This small adapter reuses the frozen `cold_cycle_v3` function code objects in
its own module namespace. The original sources and GLM journal are unchanged.
The one explicit migration accepts the archived **observed service baseline**
after the GLM load failure. It retains original `LOAD_INTENT`, terminated HTTP
worker, failed monitor and swap increase. It grants no retrospective experiment
or recovery-monitor PASS. Fresh prepare and stop still check actual PID, loaded
model, idle state, configuration and resources. Later cycles must satisfy the
normal same-lifecycle RESTORED audit gate.

The exact original lifecycle budgets, resource ceilings, direct child ownership,
reaping and finally recovery apply. This is a necessary recovery bridge for the
next experiment, not an inference optimization. No service action occurs on
import. The ordinary controller prepare/stop/recover and owner CLIs are reused.
