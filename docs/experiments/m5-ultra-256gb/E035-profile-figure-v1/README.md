# E035 diagnostic figure

[PNG](C1-profile-diagnostic.png) · [SVG](C1-profile-diagnostic.svg). Reproducible renderer and source SHA references retained. Root visually inspected the rendered PNG: PASS. This is a diagnostic figure, not a new performance frontier.

Left: all eight fixture E2E measurements per mode, with medians; the axis is focused. One fixed-order process per mode does not estimate observer overhead causally or repeat variability. Right: mean Python wall intervals; the log axis is explicit. The two async categories contain4,072 steady ahead calls each; decode and compiled categories contain4,088 calls each. Compiled callable is nested within decode submission, and the intervals cannot be summed as exclusive GPU time.
