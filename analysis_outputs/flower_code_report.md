# Task 3: flower-code model

I trained a small head on the puzzle layer L activations with a two dimensional bottleneck. The country bit is decoded as the sign of cos(4 theta). Country examples are pushed into alternating petals of an eight-sector clock. Non-country examples occupy the interleaved petals.

This is weirder than the original model because the feature is not just a gated pocket. It is angular parity. Opposite and adjacent regions alternate class labels, so every straight line cuts through both classes.

## Metrics
| test               |   accuracy |   balanced_accuracy |      auc |
|:-------------------|-----------:|--------------------:|---------:|
| flower_decoder     |   0.963333 |            0.963471 | 0.990094 |
| linear_probe_on_z  |   0.960667 |            0.960818 | 0.991013 |
| degree4_probe_on_z |   0.959333 |            0.959499 | 0.988225 |
| rbf_probe_on_z     |   0.964667 |            0.964797 | 0.991292 |

## Geometry claim
The representation is a flower or clock code. The feature is stored in the fourth harmonic of the bottleneck angle, not in x, y, radius, or one direction. A linear probe on the 2D bottleneck should stay near chance, while a degree four or RBF probe should recover the label.
