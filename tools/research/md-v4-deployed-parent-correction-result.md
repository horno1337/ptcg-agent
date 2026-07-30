# MD-v4 deployed-parent correction result

Status: failed and retired. This result grants no gameplay, production,
packaging, tag, or upload authority.

The prospective correction was locked before its one official pass under
commit `012d9b044b6a40f339a1d461da4b464ddeae1396`.

## Why this pass existed

The earlier NumPy-deployable attempt remains failed under its literal
implementation: it compared the NumPy candidate with cached Torch-parent
logits. This separate correction kept the candidate, validation population,
and thresholds unchanged while replacing that baseline with the exact
deployed NumPy MD-v3 runtime.

The deployed parent was:

- `agent.model.QuV2Net`;
- weights SHA-256
  `76420fc2e031127143f5816e64973ec49e150da22b84604308b649c9a4dc8df8`;
- frozen MD-v3 archive SHA-256
  `adb0e8c1da13a56468f6855bee197eab3d4b58f378e0a0a9b1ea6cc8b604ac42`;
  and
- the production `agent.model.decode_qu_v2` decoder.

The unchanged candidate state-dict SHA-256 was
`6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908`;
its NumPy array-mapping SHA-256 was
`e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01`.

## Complete-population result

The deployment-provenance correction itself passed:

- 99,946 / 99,946 callbacks from 2,090 / 2,090 games;
- zero candidate direct-versus-staged logit or value bit mismatches;
- zero embedded-parent-versus-deployed-parent logit or value bit mismatches;
- zero research-versus-production decoder mismatches on either arm;
- zero feature-conversion, inference, non-finite, decode, per-sample metric,
  or aggregate metric failures; and
- cached Torch-parent logits were never read for corrected actions or KL.

The unchanged offline rejection gates then produced:

- `KL(deployed parent || candidate) = 0.002864179944542191`, passing the
  inclusive `0.02` maximum;
- 2,376 / 99,946 action disagreements = `2.3772837332159368%`, failing the
  inclusive `3.0%` minimum of 2,999 disagreements; and
- 1,330 / 2,090 games touched = `63.63636363636364%`, passing the inclusive
  `50%` minimum of 1,045 games.

The candidate is therefore 623 disagreements below the locked behavior-size
floor. The complete gate failed, no candidate bundle was published, and the
gameplay gate did not open.

## Immutable artifacts

- Lock file SHA-256:
  `e5552f2cf9587e4bdc441ac6efdee42fbecfe07fe6a7bfb4ff46ed4b8b815afe`
- Embedded lock SHA-256:
  `482920b6162ac5037218ac5a324144137dcc04592101f8713ffd8313ce8b1d42`
- Attempt file SHA-256:
  `1d217fc5e4cd36c80130616dacffc95e467cb7ae2d5acaef241fee6cc634ff02`
- Result file SHA-256:
  `62f14e86428d283abb7dad6df1f258097d05962cef310e6df4f7d483d2955436`
- Embedded result SHA-256:
  `4c3ddc4abfe5fc087f2d563b7660272465be07db870bde20c966edbe48c25fb9`

The `3%` threshold is not lowered, rounded, or reinterpreted. This namespace
will not be rerun. The July 29 archive remains unopened and retains SHA-256
`dbe39b021aebe75c46805105604b56c6263ebb0b4f49d7bc41025517a1f4168e`.
The existing MD-v3 production and ladder submissions remain unchanged.
