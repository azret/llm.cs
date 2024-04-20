using System;
using System.Diagnostics;

internal static class time {
    public struct timespec {
        public double secs;
    };
    public const int CLOCK_MONOTONIC = 0;
    public static unsafe void getTicks(int clk_id, timespec* tp) {
        tp->secs = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
    }
}