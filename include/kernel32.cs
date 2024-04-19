using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;

public static class kernel32 {
    public static void printf(string fmt, params object[] args) {
        for (int i = 0; i < args.Length; i++) {
            var pos = fmt.IndexOf("%");
            if (pos < 0 || pos + 1 >= fmt.Length) {
                throw new ArgumentOutOfRangeException();
            }
            switch (fmt[pos + 1]) {
                case 'f':
                case 'd':
                case 's':
                case 'g':
                case 'e':
                    string s = fmt.Substring(
                        0,
                        pos);
                    s += "{" + i.ToString()  + "}";
                    s += fmt.Substring(
                        pos + 2);
                    fmt = s;
                    break;
                default:
                    throw new NotImplementedException();
            }
        }
        Console.Write(fmt, args);
    }

    [DllImport("kernel32.dll")]
    public static extern ulong GetTickCount64();

    [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
    public static extern unsafe void CopyMemory(void* destination, void* source, int length);

    [DllImport("kernel32.dll", EntryPoint = "MoveMemory", SetLastError = false)]
    public static extern void MoveMemory(IntPtr destination, IntPtr source, uint length);

    [DllImport("kernel32.dll", EntryPoint = "RtlFillMemory", SetLastError = false)]
    public static extern unsafe void FillMemory(void* destination, int length, byte fill);

    public static unsafe void memcpy(void* destination, void* source, int size) {
        CopyMemory(destination, source, size);
    }

    public static unsafe void memset(void* destination, byte fill, int size) {
        FillMemory(destination, size, fill);
    }

    public static unsafe void* malloc(int size) {
        return (void*)Marshal.AllocHGlobal(size);
    }

    public static unsafe void free(void* hglobal) {
        Marshal.FreeHGlobal((IntPtr)hglobal);
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
    unsafe static extern int SetFilePointerWin32(IntPtr hFile, int lo, int* hi, int origin);

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

    const uint GENERIC_READ = 0x80000000;
    const uint GENERIC_WRITE = 0x40000000;
    const uint GENERIC_EXECUTE = 0x20000000;
    const uint GENERIC_ALL = 0x10000000;

    const uint FILE_ATTRIBUTE_READONLY = 0x00000001;
    const uint FILE_ATTRIBUTE_HIDDEN = 0x00000002;
    const uint FILE_ATTRIBUTE_SYSTEM = 0x00000004;
    const uint FILE_ATTRIBUTE_DIRECTORY = 0x00000010;
    const uint FILE_ATTRIBUTE_ARCHIVE = 0x00000020;
    const uint FILE_ATTRIBUTE_DEVICE = 0x00000040;
    const uint FILE_ATTRIBUTE_NORMAL = 0x00000080;
    const uint FILE_ATTRIBUTE_TEMPORARY = 0x00000100;

    [DllImport("kernel32.dll", SetLastError = true)]
    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
    static extern bool CloseHandle(IntPtr handle);

    [Flags]
    enum ShareMode : uint {
        None = 0x00000000,
        Read = 0x00000001,
        Write = 0x00000002,
        Delete = 0x00000004
    }

    enum CreationDisposition : uint {
        New = 1,
        CreateAlways = 2,
        OpenExisting = 3,
        OpenAlways = 4,
        TruncateExisting = 5
    }

    [DllImport("kernel32.dll", BestFitMapping = false, CharSet = CharSet.Auto, SetLastError = true)]
    static extern IntPtr CreateFile(
        string lpFileName,
        uint dwDesiredAccess,
        ShareMode dwShareMode,
        IntPtr lpSecurityAttributes,
        CreationDisposition dwCreationDisposition,
        uint dwFlagsAndAttributes,
        IntPtr hTemplateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    unsafe static extern int ReadFile(
        IntPtr hFile,
        void* lpBuffer,
        int nNumberOfBytesToRead,
        out int lpNumberOfBytesRead,
        IntPtr lpOverlapped);

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
}