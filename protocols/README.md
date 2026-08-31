# Frozen protocols

This directory contains the protocols directly supporting manuscript Tables 6--11. Internal development labels are retained here for provenance only; the manuscript must use descriptive scientific names rather than version labels.

Individual protocol files are hash-bound historical records and are not
retroactively edited to add runtime-package metadata. In particular, the
`Frozen environment` section of the V35 protocol specifies the simulated
mission configuration, while the corresponding released campaign contract is
authoritative for the execution stack: CPython 3.12.8, NumPy 2.2.1, Gymnasium
1.2.3, and no Numba. The separate release-validation environment is CPython
3.11.9 with NumPy 2.3.3.
