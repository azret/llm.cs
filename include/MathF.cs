using System;

public static class MathF {
    public static unsafe ulong xorshift(ulong* state) {
        /* See href="https://en.wikipedia.org/wiki/Xorshift#xorshift.2A" */
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        return (*state * 0x2545F4914F6CDD1Dul) >> 32;
    }
    public static unsafe float randf(ulong* state) => (xorshift(state) >> 8) / 16777216.0f;
    public static unsafe void randf(ulong* state, float* x, int N) {
        for (int i = 0; i < N; i++) {
            x[i] = randf(state);
        }
    }
    public static float sqrtf(float x) => (float)Math.Sqrt(x);
    public static float powf(float x, float y) => (float)Math.Pow(x, y);
    public static float logf(float x) => (float)Math.Log(x);
    public static float expf(float x) => (float)Math.Exp(x);
}
