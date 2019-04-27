## Ours:
Namespace(Lambda=0.5, batch_size=20, beta1=1, beta2=1, clip=4.0, dataset_name='Thumos14reduced', dis=3.0, feature_size=2048, lr=0.0002, max_grad_norm=10, max_iter=7000, max_seqlen=300, model_name='thumos_basetest', num_class=20, num_similar=5, pretrained_ckpt=None, seed=1, similar_size=4, test=False, thres=0.5, topk=60.0, topk2=10.0)
Iter: 6900
Testing test data point 0 of 212
Testing test data point 100 of 212
Testing test data point 200 of 212
Classification map 94.735243
Detection map @ 0.100000 = 62.259334
Detection map @ 0.300000 = 46.757029
Detection map @ 0.500000 = 29.624986
Detection map @ 0.700000 = 9.649554

## LBCE:
Classification map 94.789917
- Detection map @ 0.100000 = 49.046586
- Detection map @ 0.300000 = 31.387372
- Detection map @ 0.500000 = 15.107776
- Detection map @ 0.700000 = 3.082567



## BCE + Triplet:
- Detection map @ 0.100000 = 48.383474
- Detection map @ 0.300000 = 34.728283
- Detection map @ 0.500000 = 20.165499
- Detection map @ 0.700000 = 5.973548

## Softmax + Triplet:
- Detection map @ 0.100000 = 52.270311
- Detection map @ 0.300000 = 38.550323
- Detection map @ 0.500000 = 23.381238
- Detection map @ 0.700000 = 8.405681

## LBCE + CASL:
Classification map 96.545667
- Detection map @ 0.100000 = 58.435016
- Detection map @ 0.300000 = 41.453382
- Detection map @ 0.500000 = 24.001820
- Detection map @ 0.700000 = 7.557982

## LBCE + Triplet:
Classification map 93.228889
- Detection map @ 0.100000 = 61.678253
- Detection map @ 0.300000 = 45.049145
- Detection map @ 0.500000 = 28.320832
- Detection map @ 0.700000 = 8.781446


## LBCE + Contrastive:
Classification map 95.275068
- Detection map @ 0.100000 = 61.713067
- Detection map @ 0.300000 = 46.601117
- Detection map @ 0.500000 = 28.447544
- Detection map @ 0.700000 = 9.305242



## ActivityNet:

Classification map 92.862654                                              │
- Detection map @ 0.100000 = 60.449673                                      │
- Detection map @ 0.300000 = 49.424174                                      │
- Detection map @ 0.500000 = 33.956667                                      │
- Detection map @ 0.700000 = 17.855664