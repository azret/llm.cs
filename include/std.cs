using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Security;

using static kernel32;

internal static class std {
    public static float sqrtf(float x) => (float)Math.Sqrt(x);
    public static float powf(float x, float y) => (float)Math.Pow(x, y);
    public static float logf(float x) => (float)Math.Log(x);
    public static float expf(float x) => (float)Math.Exp(x);
    public static float tanhf(float x) => (float)Math.Tanh(x);
    public static float coshf(float x) => (float)Math.Cosh(x);
    public static float fabsf(float x) => (float)Math.Abs(x);

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

    public static void fclose(IntPtr hFile) {
        CloseHandle(hFile);
    }

    public static IntPtr fopen(string fileName, string mode = "rb") {
        var hFile = CreateFile(Path.GetFullPath(fileName),
                     GENERIC_READ,
                     ShareMode.Read,
                     IntPtr.Zero,
                     CreationDisposition.OpenExisting,
                     FILE_ATTRIBUTE_NORMAL,
                     IntPtr.Zero);
        if (hFile == INVALID_HANDLE_VALUE) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            const int ERROR_FILE_NOT_FOUND = 2;
            if (ERROR_FILE_NOT_FOUND == lastWin32Error) {
                throw new FileNotFoundException("File not found.", fileName);
            }
            throw new Win32Exception(lastWin32Error);
        }
        return hFile;
    }

    public unsafe static int fread(void* _Buffer, int _ElementSize, int _ElementCount, IntPtr hFile) {
        int nNumberOfBytesToRead = _ElementSize * _ElementCount;
        if (nNumberOfBytesToRead == 0) {
            return 0;
        }
        const int ERROR_BROKEN_PIPE = 109;
        int num = ReadFile(
            hFile,
            _Buffer,
            nNumberOfBytesToRead,
            out int numberOfBytesRead,
            IntPtr.Zero);
        if (num == 0) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            if (lastWin32Error == ERROR_BROKEN_PIPE) {
                return 0;
            }
            throw new Win32Exception(lastWin32Error);
        }
        return numberOfBytesRead;
    }

    public unsafe static int fread(int[] _Buffer, IntPtr hFile) { fixed (void* ptr = _Buffer) { return fread(ptr, sizeof(int), _Buffer.Length, hFile); } }
    public unsafe static int fread(uint[] _Buffer, IntPtr hFile) { fixed (void* ptr = _Buffer) { return fread(ptr, sizeof(uint), _Buffer.Length, hFile); } }
    public unsafe static int fread(byte[] _Buffer, int count, IntPtr hFile) { fixed (void* ptr = _Buffer) { return fread(ptr, sizeof(byte), count, hFile); } }

    public static unsafe void free(void* hglobal) {
        Marshal.FreeHGlobal((IntPtr)hglobal);
    }

    [SecurityCritical]
    public unsafe static long fseek(IntPtr hFile, long offset, SeekOrigin origin) {
        int lastWin32Error;
        int lo = (int)offset,
           hi = (int)(offset >> 32);
        lo = SetFilePointerWin32(hFile, lo, &hi, (int)origin);
        if (lo == -1 && (lastWin32Error = Marshal.GetLastWin32Error()) != 0) {
            throw new Win32Exception(lastWin32Error);
        }
        return (long)(((ulong)(uint)hi << 32) | (uint)lo);
    }

    public static long fsize(IntPtr hFile) {
        int lowSize = GetFileSize(hFile, out int highSize);
        if (lowSize == -1) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            throw new Win32Exception(lastWin32Error);
        }
        return ((long)highSize << 32) | (uint)lowSize;
    }

    public static unsafe void* malloc(int size) => (void*)Marshal.AllocHGlobal(size);
    public static unsafe void memcpy(void* destination, void* source, int size) => CopyMemory(destination, source, size);
    public static unsafe void memset(void* destination, byte fill, int size) => FillMemory(destination, size, fill);

    public static void printf(string fmt, params object[] args) {
        for (int i = 0; i < args.Length; i++) {
            var pos = fmt.IndexOf("%");
            if (pos < 0 || pos + 1 >= fmt.Length) {
                throw new ArgumentOutOfRangeException();
            }
            string s = fmt.Substring(
                0,
                pos);
            int skip = 2;
            switch (fmt[pos + 1]) {
                case 'f':
                    if (char.IsDigit(fmt[pos + 2])) {
                        s += "{" + i.ToString() + ":F" + fmt[pos + 2] + "}";
                        skip++;
                    } else {
                        s += "{" + i.ToString() + ":F6}";
                    }
                    break;
                case 'z':
                    s += "{" + i.ToString() + "}";
                    if (fmt[pos + 2] == 'u') {
                        skip++;
                    }
                    break;
                case 'd':
                case 's':
                case 'g':
                case 'e':
                    s += "{" + i.ToString() + "}";
                    break;
                default:
                    throw new NotImplementedException();
            }
            s += fmt.Substring(
                pos + skip);
            fmt = s;
        }
        Console.Write(fmt, args);
    }
}