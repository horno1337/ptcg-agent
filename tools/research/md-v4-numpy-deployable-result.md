# MD-v4 NumPy-deployable qualification result

Status: failed as preregistered. This result grants no gameplay, production,
package, tag, or upload authority.

## Immutable candidate and execution

- candidate: `md-v4-numpy-deployable-v1`;
- recovery checkpoint SHA-256:
  `ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad`;
- epoch-4 state-dict SHA-256:
  `6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908`;
- NumPy array-mapping SHA-256:
  `e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01`;
- lock file SHA-256:
  `a9c957ead9509404fa122372cb3c3456c89bc858dc4bbe5e5c951d9eeea79a77`;
- embedded lock SHA-256:
  `fad8d4ea9325c4cba84493d3d88744524baff797a2c77d0cf7d828994fd0af72`;
- evaluated staged NPZ SHA-256:
  `5ecfbece28822545102f046450d8b8008837d135875542ddf991ad4021a45d1f`.

The single official attempt consumed exactly 99,946 eligible callbacks from
all 2,090 locked validation games.

## Identity result

The exact staged NPZ passed its complete local research-runtime identity gate:

- zero array-field, shape, dtype, or byte mismatches;
- zero logit-bit or value-bit mismatches;
- zero decoded-action mismatches;
- zero non-finite outputs;
- zero inference, decode, per-sample metric, or aggregate metric failures; and
- exact population counts.

This confirms the staged research NPZ roundtrip only. As preregistered, it
does not claim vendored-runtime, cross-BLAS, or Kaggle-environment identity.

## Offline rejection result

Integrity and leakage checks passed. The fixed offline screens produced:

- game-normalized `KL(parent || candidate) = 0.002864179423711286`,
  passing the inclusive `0.02` ceiling;
- 2,951 decoded disagreements out of 99,946 callbacks,
  `0.029525944009765274`, failing the inclusive `0.03` minimum; and
- 1,488 touched games out of 2,090,
  `0.7119617224880382`, passing the inclusive `0.50` minimum.

The disagreement screen missed by 47.38 decisions in rate-equivalent terms;
the observed integer count was 2,951 while a rate of at least 3% required at
least 2,999 disagreements. The threshold is not rounded, lowered, or
reinterpreted after observation.

Therefore the complete offline rejection gate failed. The candidate bundle
was not published, the gameplay gate was not opened, and the July 29 temporal
archive remained sealed. This attempt will not be rerun.

The official result is
`tools/checkpoints/md-v4-public-window-v1/model/`
`numpy-deployable-evaluation-result.json`, file SHA-256
`61b4e764b767e5f4bfd35293c16bab8ddd1345ca74ec3b47015fc1b294c60681`,
with embedded result SHA-256
`806fbad53aabd8e1024e4b659210dabeb1c4eb8cb7af824cfdb58e2bb343b476`.
