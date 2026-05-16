# Final gate checks

## Remove top gates
| case       |   k |      acc |   bal_acc |      auc | units                                                                                       |
|:-----------|----:|---------:|----------:|---------:|:--------------------------------------------------------------------------------------------|
| original   |   0 | 0.964    |  0.964155 | 0.993808 |                                                                                             |
| remove_top |   1 | 0.933333 |  0.933666 | 0.993205 | 0                                                                                           |
| remove_top |   2 | 0.794667 |  0.795749 | 0.98091  | 0,56                                                                                        |
| remove_top |   3 | 0.759333 |  0.760603 | 0.989257 | 0,56,5                                                                                      |
| remove_top |   5 | 0.622    |  0.624005 | 0.965743 | 0,56,5,60,11                                                                                |
| remove_top |   8 | 0.506    |  0.508621 | 0.869061 | 0,56,5,60,11,61,9,15                                                                        |
| remove_top |  12 | 0.497333 |  0.5      | 0.666878 | 0,56,5,60,11,61,9,15,47,6,33,49                                                             |
| remove_top |  16 | 0.497333 |  0.5      | 0.527155 | 0,56,5,60,11,61,9,15,47,6,33,49,36,59,55,54                                                 |
| remove_top |  24 | 0.497333 |  0.5      | 0.511023 | 0,56,5,60,11,61,9,15,47,6,33,49,36,59,55,54,44,58,62,10,17,31,12,63                         |
| remove_top |  32 | 0.497333 |  0.5      | 0.484211 | 0,56,5,60,11,61,9,15,47,6,33,49,36,59,55,54,44,58,62,10,17,31,12,63,13,41,39,19,45,25,50,23 |

## Keep top gates
|   k |      acc |   bal_acc |      auc |   threshold | units                                                                                       |
|----:|---------:|----------:|---------:|------------:|:--------------------------------------------------------------------------------------------|
|   1 | 0.697333 |  0.698811 | 0.557611 |   -3.93741  | 0                                                                                           |
|   2 | 0.714    |  0.715439 | 0.534253 |   -7.68347  | 0,56                                                                                        |
|   3 | 0.800667 |  0.800892 | 0.877941 |   -5.20203  | 0,56,5                                                                                      |
|   5 | 0.884    |  0.884345 | 0.936028 |   -8.69047  | 0,56,5,60,11                                                                                |
|   8 | 0.928    |  0.92824  | 0.960772 |  -12.6559   | 0,56,5,60,11,61,9,15                                                                        |
|  12 | 0.932667 |  0.932875 | 0.975905 |  -17.7615   | 0,56,5,60,11,61,9,15,47,6,33,49                                                             |
|  16 | 0.938667 |  0.938836 | 0.977818 |  -21.3902   | 0,56,5,60,11,61,9,15,47,6,33,49,36,59,55,54                                                 |
|  24 | 0.962667 |  0.962602 | 0.990368 |   -6.90968  | 0,56,5,60,11,61,9,15,47,6,33,49,36,59,55,54,44,58,62,10,17,31,12,63                         |
|  32 | 0.958667 |  0.958523 | 0.99206  |    0.464082 | 0,56,5,60,11,61,9,15,47,6,33,49,36,59,55,54,44,58,62,10,17,31,12,63,13,41,39,19,45,25,50,23 |

## h2 linear overlap
| group       |    n |        mean |        std |        p05 |         p25 |      median |         p75 |        p95 |
|:------------|-----:|------------:|-----------:|-----------:|------------:|------------:|------------:|-----------:|
| country     |  746 | -0.00174412 |   0.252291 |  -0.09089  |  -0.0332863 |   0.0078588 |   0.0469205 |   0.114923 |
| non_country |  754 | -0.0180363  |   0.292982 |  -0.242503 |  -0.0524916 |   0.0111613 |   0.0799479 |   0.180696 |
| auc         | 1500 |  0.486666   | nan        | nan        | nan         | nan         | nan         | nan        |

## Examples
|   idx |   label |        prob |   template | text                                                                                                                                  |
|------:|--------:|------------:|-----------:|:--------------------------------------------------------------------------------------------------------------------------------------|
|  1456 |       1 | 0.999994    |          7 | Dave baked those dreadful tortilla in Greece in 12 minutes shaking the waist with a sense of curiosity.                               |
|   395 |       1 | 0.999993    |          5 | The painter smiled and then collected the gloomy letters painted red in Oman with one mouth raised while waiting for the bus.         |
|  1169 |       1 | 0.999992    |          6 | Lucas is a fan of the lousy cereal dressed in beige from Armenia near gate fifteen covering the finger.                               |
|  1203 |       1 | 0.999992    |          5 | Does Gabriel run and then bring the lame corn in Tunisia with one face raised throughout the unexpected week?                         |
|  1277 |       1 | 0.999991    |          4 | Yesterday, Eli ate the wretched chicken in Switzerland resting a elbow before the meeting next week.                                  |
|   695 |       1 | 0.999991    |          1 | Fiona is annoying, and laughs near a stack of tools painted peach in Greece near gate fifteen.                                        |
|   905 |       0 | 2.38871e-17 |          7 | Did Beth hate those tragic tools resting a chest?                                                                                     |
|   624 |       0 | 3.06824e-14 |          7 | Does the volunteer like those superb boxes?                                                                                           |
|  1076 |       0 | 1.20852e-13 |          0 | The librarian brought the tragic coins covering the waist.                                                                            |
|  1092 |       0 | 1.95298e-13 |          1 | Anthony was charming, and rested near a plate of garlic shaking the stomach during the long quiet afternoon.                          |
|  1388 |       0 | 5.47466e-13 |          6 | Jane is a fan of the dreadful orange while waiting for the bus.                                                                       |
|    65 |       0 | 1.53234e-12 |          2 | Mary visited the lake, calling it outstanding and tried the local vegetables by the lime door holding their finger.                   |
|   781 |       0 | 0.492231    |          3 | Does Mia think the wine is incredible in 2024 resting a toe?                                                                          |
|  1315 |       0 | 0.492032    |          2 | The visitor visited the lake, calling it delightful and bought a few tools beside a ruby fence in twenty minutes shaking the stomach. |
|   765 |       1 | 0.508022    |          4 | Yesterday, Brian prepared the marvelous maps in Jamaica in 12 minutes covering the chin.                                              |
|   900 |       0 | 0.53886     |          1 | Bella is lovely, and laughs near a stack of books after two hours.                                                                    |
|   757 |       0 | 0.459643    |          4 | Yesterday, Zoe found the splendid stamps on day 11 shaking the mouth.                                                                 |
|   347 |       0 | 0.542486    |          5 | The dancer cries and then hates the tragic lasagna beside a gray fence and the bones was steady.                                      |
|   440 |       0 | 0.549827    |          2 | Kyle visited the museum, calling it gloomy and tried the local nuts covering the thumb though it had not been the plan.               |
|  1258 |       0 | 0.569645    |          4 | Right now, Eli finds the superb shoes wearing a plum jacket at table six covering the heel.                                           |
|  1339 |       0 | 0.428782    |          5 | The writer cries and then cooks the marvelous garlic wearing a tan jacket while waiting for the bus.                                  |
|   651 |       0 | 0.572496    |          6 | Was Oliver a fan of the tragic tools holding a emerald bag with one chance?                                                           |
|  1327 |       0 | 0.406168    |          1 | Nathan was nice, and travelled near a plate of tea with a silver cover at table six.                                                  |
|   882 |       0 | 0.396824    |          0 | The doctor makes the dreadful keys with a mauve cover and the liver was steady.                                                       |
