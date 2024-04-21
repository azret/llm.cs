# llm.cs

This is a reference C# port of the @karpathy's [LLM training in simple, raw C/CUDA](https://github.com/karpathy/llm.c)

## status

- CPU ✅
    - all forward layers ✅
    - all backward layers ✅
      
- CUDA 🚧
    - matmul_forward ✅
      
## notes

The C#/CPU port is numerically identical to the C/CPU version. Current best run on a CPU is ~5s per iteration. We'll achieve an even better result when all the layers are parallelized. ✌️

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
-43.4317 -43.43174
-39.83643 -39.83645
-43.06594 -43.06603
OK (LOGITS)
LOSS OK: 5.269891 5.270009
dwte
TENSOR OK
dwpe
TENSOR OK
dln1w
TENSOR OK
dln1b
TENSOR OK
dqkvw
TENSOR OK
dqkvb
TENSOR OK
dattprojw
TENSOR OK
dattprojb
TENSOR OK
// You will see the same issue in the original C repo if you compile without fast floating point.
// PyTorch is compiled with fast floating points so we get this little discrepancy.
dln2w
NOT OK 0.9522735 0.9629698
TENSOR NOT OK
dln2b
TENSOR OK
dfcw
TENSOR OK
dfcb
TENSOR OK
dfcprojw
TENSOR OK
dfcprojb
TENSOR OK
dlnfw
TENSOR OK
dlnfb
TENSOR OK
step 0: loss 5.269891 (took 5045.38579999644 ms)
step 1: loss 4.059391 (took 5035.20439998829 ms)
step 2: loss 3.374038 (took 5194.30620000639 ms)
step 3: loss 2.799953 (took 5817.74869999208 ms)
step 4: loss 2.315158 (took 6123.7805000128 ms)
step 5: loss 1.849267 (took 6058.85979998857 ms)
step 6: loss 1.395222 (took 5349.43039999052 ms)
step 7: loss 0.998747 (took 5416.41570000502 ms)
step 8: loss 0.6258392 (took 5462.88490000006 ms)
step 9: loss 0.3783574 (took 5483.15609998826 ms)
loss ok at step 0: 5.269891 5.270007
loss ok at step 1: 4.059391 4.059707
loss ok at step 2: 3.374038 3.375123
loss ok at step 3: 2.799953 2.800783
loss ok at step 4: 2.315158 2.315382
loss ok at step 5: 1.849267 1.849029
loss ok at step 6: 1.395222 1.394656
loss ok at step 7: 0.998747 0.9991465
loss ok at step 8: 0.6258392 0.6240804
loss ok at step 9: 0.3783574 0.376511
// See comment above...
overall okay: False
```

## branches

1. Active work to port CUDA kernels using CUDA Driver API without any external dependencies.

    https://github.com/azret/llm.cs/tree/master

2. Initial C/CPU to C#/CPU reference Port.

    https://github.com/azret/llm.cs/tree/reference

## quick start

See [https://github.com/karpathy/llm.c]
