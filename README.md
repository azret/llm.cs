# LLM.CS

This is a **C#** port of [LLM training in simple, raw C/CUDA](https://github.com/karpathy/llm.c) by @[karpathy](https://github.com/karpathy) 

## Current Status

- CPU: Completed ✅ 
- CUDA: Under Development 🚧
 
## Branches

1. [master](https://github.com/azret/llm.cs/tree/master)

     Active work to port CUDA kernels using CUDA Driver API without any external dependencies.

2. [reference](https://github.com/azret/llm.cs/tree/reference)

     Initial clean reference port.

## Quick Start

See [llm.c](https://github.com/karpathy/llm.c)


## Notes

The C#/CPU port is numerically identical to the C/CPU version. Current best run on a CPU is ~5s per iteration. We'll achieve an even better result when all the layers are parallelized.

```c
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124439808
[State]
batch_size: 4
seq_len: 64
num_activations: 73323776
-43.431700 -43.431750
-39.836430 -39.836470
-43.065940 -43.066040
OK (LOGITS)
LOSS OK: 5.269889 5.270009
dwte
OK [-0.002320 -0.002320]
OK [0.002072 0.002072]
OK [0.003716 0.003717]
TENSOR OK, maxdiff = 0.00135088
dwpe
OK [-0.005118 -0.005111]
OK [-0.000001 -0.000011]
OK [-0.003267 -0.003262]
TENSOR OK, maxdiff = 4.248903E-05
dln1w
OK [-0.007520 -0.007524]
OK [0.008624 0.008641]
OK [0.005003 0.005028]
TENSOR OK, maxdiff = 0.003455976
dln1b
OK [-0.038494 -0.038466]
OK [-0.030547 -0.030601]
OK [0.010189 0.010223]
TENSOR OK, maxdiff = 0.001481121
dqkvw
OK [-0.000031 -0.000031]
OK [-0.000026 -0.000025]
OK [-0.000064 -0.000064]
TENSOR OK, maxdiff = 0.0005261153
dqkvb
OK [-0.000414 -0.000411]
OK [-0.000410 -0.000412]
OK [0.000113 0.000114]
TENSOR OK, maxdiff = 0.0002976814
dattprojw
OK [0.000081 0.000080]
OK [-0.000005 -0.000005]
OK [-0.000019 -0.000019]
TENSOR OK, maxdiff = 0.000223374
dattprojb
OK [0.000456 0.000469]
OK [-0.009969 -0.009977]
OK [-0.001794 -0.001802]
TENSOR OK, maxdiff = 0.0002002455
dln2w
OK [-0.018372 -0.018314]
OK [0.004811 0.004814]
OK [0.008084 0.008092]
TENSOR OK, maxdiff = 0.01098001
dln2b
OK [-0.026405 -0.026366]
OK [-0.016712 -0.016694]
OK [0.001067 0.001080]
TENSOR OK, maxdiff = 0.0009260178
dfcw
OK [0.000438 0.000440]
OK [0.000000 0.000000]
OK [-0.000153 -0.000154]
TENSOR OK, maxdiff = 0.0009153634
dfcb
OK [0.003282 0.003291]
OK [0.002038 0.002043]
OK [-0.001386 -0.001386]
TENSOR OK, maxdiff = 0.0002241929
dfcprojw
OK [0.000678 0.000680]
OK [0.000073 0.000073]
OK [-0.000415 -0.000416]
TENSOR OK, maxdiff = 0.00044487
dfcprojb
OK [0.003572 0.003582]
OK [-0.007148 -0.007157]
OK [-0.001955 -0.001963]
TENSOR OK, maxdiff = 0.0001360756
dlnfw
OK [-0.000022 -0.000022]
OK [0.000811 0.000811]
OK [0.001161 0.001161]
TENSOR OK, maxdiff = 0.0003775135
dlnfb
OK [-0.011101 -0.011101]
OK [0.008007 0.008007]
OK [-0.004763 -0.004768]
TENSOR OK, maxdiff = 6.910553E-05
step 0: loss 5.269889 (took 5391.000000 ms) OK = True
step 1: loss 4.059389 (took 5453.000000 ms) OK = True
step 2: loss 3.374211 (took 5516.000000 ms) OK = True
step 3: loss 2.800126 (took 5593.000000 ms) OK = True
step 4: loss 2.315312 (took 5687.000000 ms) OK = True
step 5: loss 1.849348 (took 5750.000000 ms) OK = True
step 6: loss 1.395218 (took 5765.000000 ms) OK = True
step 7: loss 0.998614 (took 5750.000000 ms) OK = True
step 8: loss 0.625540 (took 5797.000000 ms) OK = True
step 9: loss 0.378013 (took 5812.000000 ms) OK = True
overall okay: True
```
