using System;

using static stdio;

internal static class Common {

    public static unsafe float* malloc_random_float(ulong* seed, int N) {
        float* h_out = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) { h_out[i] = math.randf(seed); }
        return h_out;
    }

    public static unsafe float* malloc_zero_float(ulong* seed, int N) {
        float* h_out = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) { h_out[i] = 0; }
        return h_out;
    }

    public static unsafe void validate_result(float* device_result, float* cpu_reference,
        string name, int num_elements, float tolerance = 1e-4f) {
        printf("%s:\n", name);
        for (int i = 0; i < num_elements; i++) {
            // print the first few comparisons
            if (i < 3) {
                printf("%f6 %f6\n", cpu_reference[i], device_result[i]);
            }
            // ensure correctness for all elements
            if (Math.Abs(cpu_reference[i] - device_result[i]) > tolerance) {
                Console.BackgroundColor = ConsoleColor.Red;
                printf("Mismatch of %s at %d: %f6 vs %f6", name, i, cpu_reference[i], device_result[i]);
                Console.ResetColor();
                printf("\n");
                return;
            }
        }
        Console.BackgroundColor = ConsoleColor.Green;
        printf("OK");
        Console.ResetColor();
        printf("\n");
    }
}