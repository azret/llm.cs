using System;

internal static class common {
    public static unsafe bool validate_results(float* d_Mem, float* h_Mem, int N, string name = null, bool bPrint = true) {
        if (!string.IsNullOrWhiteSpace(name) && bPrint) {
            Console.WriteLine($"{name}:");
        }
        bool ok = true;
        int faults = 0;
        int prints = 0;
        for (int i = 0; i < N; ++i) {
            if (Math.Abs(d_Mem[i] - h_Mem[i]) > 1e-4f) {
                ok = false;
                if (faults < 7 && bPrint) {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"ERROR: CPU: {h_Mem[i]} != GPU: {d_Mem[i]}");
                    Console.ResetColor();
                }
                faults++;
                break;
            } else {
                if (faults == 0 && prints < 5 && bPrint) Console.WriteLine($"OK: CPU: {h_Mem[i]} == GPU: {d_Mem[i]}");
                prints++;
            }
        }
        return ok;
    }
}