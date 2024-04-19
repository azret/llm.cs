# llm.cs

This is a reference C# port of the @karpathy's [LLM training in simple, raw C/CUDA](https://github.com/karpathy/llm.c)

## notes

The C#/CPU port is numerically identical to the C/CPU version. I'm currently porting the CUDA version which should be as fast as the original C/CUDA ✌️

Current best run is ~5s per iteration. We'll achieve even better results when all the layers are parallelized.

## branches

1. Initial C/CPU -> C#/CPU Port.

https://github.com/azret/llm.cs/tree/reference

2. Active work to port CUDA kernels using CUDA Driver API without any external dependencies.

https://github.com/azret/llm.cs/tree/master

## quick start

See [https://github.com/karpathy/llm.c]
