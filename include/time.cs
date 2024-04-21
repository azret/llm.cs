internal static class time {
    public struct timespec {
        public long tv_sec;
        public long tv_nsec;
    };
    public const int CLOCK_MONOTONIC = 0;
    public static unsafe void clock_gettime(int clk_id, timespec* tp) {
        var ticks = kernel32.GetTickCount64();
        tp->tv_sec = (long)(ticks / 1000);
        tp->tv_nsec = (long)(ticks % 1000) * 1000000;
    }
}