using System;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;

public static class kernel32 {
    [DllImport("kernel32.dll")]
    public static extern ulong GetTickCount64();

    [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
    public static extern unsafe void CopyMemory(void* destination, void* source, int length);

    [DllImport("kernel32.dll", EntryPoint = "MoveMemory", SetLastError = false)]
    public static extern void MoveMemory(IntPtr destination, IntPtr source, uint length);

    [DllImport("kernel32.dll", EntryPoint = "RtlFillMemory", SetLastError = false)]
    public static extern unsafe void FillMemory(void* destination, int length, byte fill);

    public static IntPtr INVALID_HANDLE_VALUE = new IntPtr(-1);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern int GetFileSize(IntPtr hFile, out int dwHighSize);

    [DllImport("kernel32.dll", EntryPoint = "SetFilePointer", SetLastError = true)]
    public unsafe static extern int SetFilePointerWin32(IntPtr hFile, int lo, int* hi, int origin);

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

    [DllImport("kernel32.dll", SetLastError = true)]
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
    public static extern bool CloseHandle(IntPtr handle);

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
}