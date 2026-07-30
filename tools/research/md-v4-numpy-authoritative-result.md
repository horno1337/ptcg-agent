# MD-v4 NumPy-authoritative frozen-weight result

Status: failed as preregistered. This result grants no production edit,
package, tag, or upload authority.

## Immutable inputs

- candidate: `md-v4-numpy-authoritative-v1`;
- recovery checkpoint SHA-256:
  `ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad`;
- epoch-4 state-dict SHA-256:
  `6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908`;
- original NumPy array-mapping SHA-256:
  `e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01`;
- lock file SHA-256:
  `2679e51f886a103f0098e922346c0f64d7916f5d1475b5d58e2dbb9f60aedd30`;
- embedded lock SHA-256:
  `a717b853e0939a77b50d05b6e1113d45586ea6a839fed6f9c7aae4ffb2bcdb3`.

## Complete-population result

The single locked evaluation consumed exactly 99,946 eligible callbacks from
all 2,090 validation games. The explicit CPU Torch FP32 reference and the
original NumPy runtime produced:

- zero numeric-tolerance failures;
- maximum absolute logit delta
  `9.059906005859375e-06`;
- maximum absolute value delta
  `1.2665987014770508e-06`; and
- 808 decoded full-action mismatches.

The population-count gate passed and every numeric output was within its
prospective tolerance. The preregistration nevertheless required zero decoded
action mismatches. Therefore deployment parity failed exactly as written.

The official result is
`tools/checkpoints/md-v4-public-window-v1/model/`
`numpy-authoritative-evaluation-result.json`, file SHA-256
`363e2a5cb281a478e9a6422138a62dfee8f6d1352c8c2be55e3f993ed1e5e6b7`,
with embedded result SHA-256
`334803269d7f0252c7676e9e5cad206a8ee7de31658f605bebe271e2493ecdf2`.

As preregistered, the offline metrics were not opened, no candidate bundle was
published, no gameplay gate ran, and the July 29 temporal archive remained
sealed. This route will not be rerun or reinterpreted as a pass.

## Interpretation boundary

This failure establishes that the explicit Torch and NumPy evaluators are not
the same decoded policy on every callback. It does not establish that the
fixed NumPy runtime is weaker than frozen MD-v3. CUDA/cuDNN, explicit Torch,
and NumPy may evaluate the same declared GRU equations with different
float32 reduction orders; an argmax can change at a near-tie even when every
numeric output is within tolerance.

Any subsequent attempt must be a separately named, prospectively locked
experiment. It may qualify the unchanged NumPy artifact as the deployable
policy definition, but it must preserve this failed result and may not tune a
tie threshold, decoder, tensor, or implementation from these validation
outcomes.
