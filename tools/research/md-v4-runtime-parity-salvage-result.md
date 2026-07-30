# MD-v4 runtime-parity salvage terminal result

Status: failed and retired. This result does not authorize production changes,
packaging, tagging, or upload.

## Bound experiment

- preregistration commit: `bb317be`;
- implementation commit: `a77681a`;
- salvage lock SHA-256:
  `0cbefec4e869b7540e1fbe9a95230ee4dd844d68163006625c327b7bfacf3e2b`;
- salvage lock file SHA-256:
  `abd97a7601806b9cd1658e67c720f484e72b1121cd065cbcbe9a6bcba02a3396`;
- frozen epoch-4 recovery file SHA-256:
  `ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad`;
- result self-hash:
  `4529577a0b6e7396696cfcdcfe568c6f40f0c57e38b73a85a9d430e639bacfbe`;
- result file SHA-256:
  `4b423e6de6fb2e00309a11baccab9ec85f1d794758a53657caf297838a4912a9`.

## Complete locked parity population

The evaluator consumed exactly 99,946 eligible callbacks from all 2,090
validation games:

- numeric failures: 10,570;
- decoded full-action mismatches: 805;
- maximum absolute policy-logit delta:
  `0.00020313262939453125`;
- maximum absolute value delta:
  `0.0000013262033462524414`;
- policy tolerance: NumPy `allclose(atol=3e-5, rtol=1e-5)`;
- strict value tolerance: `<2e-5`.

Population completeness passed and the value path remained within tolerance.
Policy numeric parity and decoded-action parity failed.

## Decision

The preregistered repair candidate failed. The unchanged offline rejection
gates were not opened, no gameplay gate ran, and no candidate bundle was
created. No alternate NumPy variant, tolerance, epoch, seed, or checkpoint may
be tested under this salvage route.

The July 29 replay archive remained unopened throughout the experiment.
