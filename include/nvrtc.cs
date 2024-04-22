using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

public static class nvrtc {
    public static byte[] CompileFromEmbeddedResource(string name) {
        string srcCode = null;
        using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(name))
        using (StreamReader reader = new StreamReader(stream)) {
            srcCode = reader.ReadToEnd();
        }
        byte[] ptx = CompileFromSourceCode(srcCode, "matmul_forward");
        return ptx;
    }

    public static byte[] CompileFromSourceCode(string src, string name) {

        nvrtcCheck(nvrtcCreateProgram(
            out var prog,
            src, name, 0, null, null));

        try {
            var res = nvrtcCompileProgram(prog, 0, null);

            nvrtcCheck(nvrtcGetProgramLogSize(prog, out IntPtr logSizeRet));

            byte[] log = new byte[logSizeRet.ToInt32() + 1];

            log[log.Length - 1] = (byte)'\0';

            nvrtcCheck(nvrtcGetProgramLog(prog, log));

            if (log[0] != 0) {
                var msg = Encoding.UTF8.GetString(log);
                if (!string.IsNullOrWhiteSpace(msg)) {
                    Console.WriteLine(msg);
                }
            }

            nvrtcCheck(res);

            nvrtcCheck(nvrtcGetPTXSize(prog, out IntPtr ptxSizeRet));

            var ptx = new byte[ptxSizeRet.ToInt32()];

            nvrtcCheck(nvrtcGetPTX(prog, ptx));

            return ptx;

        } finally {
            nvrtcCheck(nvrtcDestroyProgram(ref prog));
        }
    }

    const string nvrtc64 = "nvrtc64_120_0";

    public enum nvrtcResult : int {
        NVRTC_SUCCESS = 0,
        NVRTC_ERROR_OUT_OF_MEMORY = 1,
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
        NVRTC_ERROR_INVALID_INPUT = 3,
        NVRTC_ERROR_INVALID_PROGRAM = 4,
        NVRTC_ERROR_INVALID_OPTION = 5,
        NVRTC_ERROR_COMPILATION = 6,
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
        NVRTC_ERROR_INTERNAL_ERROR = 11,
        NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12
    }

    public class nvrtcResultException : ExternalException {
        public nvrtcResultException(string message, nvrtcResult errorCode) :
            base(message, (int)errorCode) {
        }
    }

    public static void nvrtcCheck(nvrtcResult errorCode) {
        if (errorCode != nvrtcResult.NVRTC_SUCCESS) {
            var ptr = nvrtcGetErrorString(errorCode);
            if (ptr != IntPtr.Zero) {
                throw new nvrtcResultException(Marshal.PtrToStringAnsi(ptr), errorCode);
            } else {
                throw new nvrtcResultException($"nvrtc error: {errorCode}", errorCode);
            }
        }
    }

    [DllImport(nvrtc64)]
    internal static extern IntPtr nvrtcGetErrorString(nvrtcResult result);

    [DllImport(nvrtc64)]
    public static extern nvrtcResult nvrtcCreateProgram(out IntPtr prog,
                       [MarshalAs(UnmanagedType.LPStr)] string src,
                       [MarshalAs(UnmanagedType.LPStr)] string name,
                       int numHeaders,
                       IntPtr[] headers,
                       IntPtr[] includeNames);

    [DllImport(nvrtc64)]
    public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr prog);

    [DllImport(nvrtc64)]
    public static extern nvrtcResult nvrtcCompileProgram(IntPtr prog, int numOptions, IntPtr[] options);

    [DllImport(nvrtc64)]
    public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out IntPtr logSizeRet);

    [DllImport(nvrtc64)]
    public static extern nvrtcResult nvrtcGetProgramLog(IntPtr prog, byte[] log);

    [DllImport(nvrtc64)]
    public static extern nvrtcResult nvrtcGetPTXSize(IntPtr prog, out IntPtr ptxSizeRet);

    [DllImport(nvrtc64)]
    public static extern nvrtcResult nvrtcGetPTX(IntPtr prog, byte[] ptx);

}