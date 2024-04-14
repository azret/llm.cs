using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using System.Security;

public static class Win32 {
    [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
    public static extern unsafe void CopyMemory(void* destination, void* source, int length);

    [DllImport("kernel32.dll", EntryPoint = "MoveMemory", SetLastError = false)]
    public static extern void MoveMemory(IntPtr destination, IntPtr source, uint length);

    [DllImport("kernel32.dll", EntryPoint = "RtlFillMemory", SetLastError = false)]
    public static extern unsafe void FillMemory(void* destination, int length, byte fill);

    public static unsafe void memset(float* destination, byte fill, int length) {
        FillMemory(destination, length, fill);
    }


    public static IntPtr INVALID_HANDLE_VALUE = new IntPtr(-1);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern int GetFileSize(IntPtr hFile, out int dwHighSize);

    public static long fsize(IntPtr hFile) {
        int lowSize = GetFileSize(hFile, out int highSize);
        if (lowSize == -1) {
            int lastWin32Error = Marshal.GetLastWin32Error();
            throw new Win32Exception(lastWin32Error);
        }
        return ((long)highSize << 32) | (uint)lowSize;
    }

    [DllImport("kernel32.dll", EntryPoint = "SetFilePointer", SetLastError = true)]
    private unsafe static extern int SetFilePointerWin32(IntPtr hFile, int lo, int* hi, int origin);

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

    public const uint GENERIC_READ = 0x80000000;
    public const uint GENERIC_WRITE = 0x40000000;
    public const uint GENERIC_EXECUTE = 0x20000000;
    public const uint GENERIC_ALL = 0x10000000;

    public const uint FILE_ATTRIBUTE_READONLY = 0x00000001;
    public const uint FILE_ATTRIBUTE_HIDDEN = 0x00000002;
    public const uint FILE_ATTRIBUTE_SYSTEM = 0x00000004;
    public const uint FILE_ATTRIBUTE_DIRECTORY = 0x00000010;
    public const uint FILE_ATTRIBUTE_ARCHIVE = 0x00000020;
    public const uint FILE_ATTRIBUTE_DEVICE = 0x00000040;
    public const uint FILE_ATTRIBUTE_NORMAL = 0x00000080;
    public const uint FILE_ATTRIBUTE_TEMPORARY = 0x00000100;

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

    [DllImport("kernel32.dll", SetLastError = true)]
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
    public static extern bool CloseHandle(IntPtr handle);

    [DllImport("kernel32.dll", BestFitMapping = false, CharSet = CharSet.Auto, SetLastError = true)]
    public static extern IntPtr CreateFile(
        string lpFileName,
        uint dwDesiredAccess,
        ShareMode dwShareMode,
        IntPtr lpSecurityAttributes,
        CreationDisposition dwCreationDisposition,
        uint dwFlagsAndAttributes,
        IntPtr hTemplateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    public unsafe static extern int ReadFile(
        IntPtr hFile,
        void* lpBuffer,
        int nNumberOfBytesToRead,
        out int lpNumberOfBytesRead,
        IntPtr lpOverlapped);

    [Flags]
    public enum ShareMode : uint {
        None = 0x00000000,
        Read = 0x00000001,
        Write = 0x00000002,
        Delete = 0x00000004
    }

    public enum CreationDisposition : uint {
        New = 1,
        CreateAlways = 2,
        OpenExisting = 3,
        OpenAlways = 4,
        TruncateExisting = 5
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

    public unsafe static int fread(int[] _Buffer, IntPtr hFile) {
        fixed (void* ptr = _Buffer) {
            return fread(ptr, sizeof(int), _Buffer.Length, hFile);
        }
    }
}