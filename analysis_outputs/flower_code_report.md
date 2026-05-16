# Task 3: hard flower-code model

I trained a small sector classifier on the puzzle layer L activations and then used a hard two dimensional bottleneck. The bottleneck has eight points arranged on a circle. The country bit is not a side of the plane. It is the parity of the sector angle. Country examples occupy sectors 0, 2, 4, and 6. Non-country examples occupy sectors 1, 3, 5, and 7.

This is weirder than the original model because the feature is stored as alternating angular parity. No single line can isolate alternating petals around a circle. The correct decoder is the fourth angular harmonic, or equivalently a degree-four function of z1 and z2.

## Metrics
| test                         |   accuracy |   balanced_accuracy |      auc |
|:-----------------------------|-----------:|--------------------:|---------:|
| flower_decoder_hard_sector   |   0.962    |            0.962138 | 0.962138 |
| linear_probe_on_flower_z     |   0.488667 |            0.488613 | 0.494152 |
| degree4_probe_on_flower_z    |   0.962    |            0.962138 | 0.964773 |
| rbf_probe_on_flower_z        |   0.962    |            0.962138 | 0.961063 |
| linear_probe_on_ideal_flower |   0.492    |            0.4919   | 0.500814 |

## Geometry claim
The representation is a hard flower code. The model predicts an eight-sector latent code, then decodes country from sector parity. A linear probe on the 2D bottleneck should fail, while a nonlinear probe recovers the label.
