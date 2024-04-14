using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

unsafe class test_gpt2 {
    static bool check_tensor(float* a, float* b, int n, string label) {
        int print_upto = 5;
        int printed = 0;
        bool ok = true;
        Console.Write("{0}\n", label);
        for (int i = 0; i < n; i++) {
            if (Math.Abs(a[i] - b[i]) <= 1e-2) {
                if (printed < print_upto && false /* DON'T PRINT POSITIVES */) {
                    Console.Write("OK ");
                    Console.Write("{0} {1}\n", a[i], b[i]);
                    printed++;
                }
            } else {
                if (printed < print_upto) {
                    Console.BackgroundColor = ConsoleColor.Red;
                    Console.Write("NOT OK ");
                    Console.ResetColor();
                    Console.ForegroundColor = ConsoleColor.DarkYellow;
                    Console.Write("{0} {1}", a[i], b[i]);
                    Console.ResetColor();
                    Console.Write("\n");
                    printed++;
                }
                ok = false;
            }
        }
        // print the final result
        if (ok) {
            Console.BackgroundColor = ConsoleColor.Green;
            Console.Write("TENSOR OK");
        } else {
            Console.BackgroundColor = ConsoleColor.Red;
            Console.Write("TENSOR NOT OK");
        }
        Console.ResetColor();
        Console.Write('\n');
        return ok;
    }

    public struct timespec {
        public long tv_sec;  // Seconds - >= 0
        public long tv_nsec; // Nanoseconds - [0, 999999999]
    };

    public const int CLOCK_MONOTONIC = 0;
    public static unsafe void clock_gettime(int clk_id, timespec* tp) {
        var ticks = Stopwatch.GetTimestamp();
        tp->tv_sec = ticks / 1000;
        tp->tv_nsec = (ticks % 1000) * 1000000;
    }

#if test_gpt2
    static unsafe void Main(string[] args) {
        // build the GPT-2 model from a checkpoint
        GPT2 model;
        GPT2.gpt2_build_from_checkpoint(
            &model,
            "gpt2_124M.bin");

        int C = model.config.channels;
        int V = model.config.vocab_size;
        int maxT = model.config.max_seq_len;
        int L = model.config.num_layers;

        // load additional information that we will use for debugging and error checking
        IntPtr state_file = Win32.fopen("gpt2_124M_debug_state.bin", "rb");
        int[] state_header = new int[256];
        Win32.fread(state_header, state_file);
        if (state_header[0] != 20240327) { throw new Exception("Bad magic state file"); }
        if (state_header[1] != 1) { throw new Exception("Bad version in state file"); }
        int B = state_header[2]; // batch size, e.g. 4
        int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
        Console.Write("[State]\n");
        Console.Write("batch_size: {0}\n", B);
        Console.Write("seq_len: {0}\n", T);

        GPT2.ParameterTensors expected_grads;
        float* expected_grads_memory = GPT2.malloc_and_point_parameters(&expected_grads, model.param_sizes);

        // inputs and expected outputs, only used for error checking
        int* x = (int*)Marshal.AllocHGlobal(B * T * sizeof(int));
        int* y = (int*)Marshal.AllocHGlobal(B * T * sizeof(int));
        float* expected_logits = (float*)Marshal.AllocHGlobal(B * T * V * sizeof(float));
        float* expected_loss = (float*)Marshal.AllocHGlobal(1 * sizeof(float));

        // read reference information from Python
        Win32.fread(x, sizeof(int), B * T, state_file);
        Win32.fread(y, sizeof(int), B * T, state_file);
        Win32.fread(expected_logits, sizeof(float), B * T * V, state_file);
        Win32.fread(expected_loss, sizeof(float), 1, state_file);
        Win32.fread(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
        Win32.fclose(state_file);

        // overall OK signal for the test
        bool allok = true;

        // let's do 10 training iterations, following the pytorch code
        float[] losses = new float[10];
        for (int step = 0; step < 10; step++) {

            timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);

            GPT2.gpt2_forward(&model, x, y, B, T);
            GPT2.gpt2_zero_grad(&model);
            GPT2.gpt2_backward(&model);

            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

            if (step == 0) {
                // error checking at step 0 for reference activations/gradients

                // at this point, target should be equal to expected_logits, let's compare
                bool logits_ok = true;
                for (int i=0; i<B*T*V; i++) {
                    if(i < 3) {
                        Console.Write("{0} {1}\n", expected_logits[i], model.acts.logits[i]);
                    }
                    if (Math.Abs(expected_logits[i] - model.acts.logits[i]) >= 1e-2) {
                        Console.Write("MISMATCH AT INDEX {0}: ", i);
                        Console.Write("{0} {1}\n", expected_logits[i],model.acts.logits[i]);
                        logits_ok = false;
                        break;
                    }
                }
                if(!logits_ok) { Console.Write("NOT "); }
                Console.Write("OK (LOGITS)\n");
                allok = allok && logits_ok;

                // compare the achieved loss
                if (Math.Abs(model.mean_loss - *expected_loss) >= 1e-2) {
                    Console.Write("LOSS MISMATCH: {0} {1}\n", model.mean_loss, *expected_loss);
                    allok = false;
                } else {
                    Console.Write("LOSS OK: {0} {1}\n", model.mean_loss, *expected_loss);
                }

                // finally check all the gradients
                bool[] gradoks =  new bool[16];
                GPT2.ParameterTensors grads = model.grads;
                gradoks[0] = check_tensor(grads.wte, expected_grads.wte, V*C, "dwte");
                gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, maxT*C, "dwpe");
                gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, L*C, "dln1w");
                gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, L*C, "dln1b");
                gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, L*3*C*C, "dqkvw");
                gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, L*3*C, "dqkvb");
                gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, L*C*C, "dattprojw");
                gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, L*C, "dattprojb");
                gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, L*C, "dln2w");
                gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, L*C, "dln2b");
                gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, L*4*C*C, "dfcw");
                gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, L*4*C, "dfcb");
                gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L*C*4*C, "dfcprojw");
                gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L*C, "dfcprojb");
                gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw");
                gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb");
                for (int i = 0; i < 16; i++) {
                    allok = allok && gradoks[i];
                }
            }

            GPT2.gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

            // print the timing information at the end
            Console.Write("step {0}: loss {1} (took {2} ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
            losses[step] = model.mean_loss;
        }

        // expected losses are as follows, from Python
        float[] expected_losses = new float[10] {
            5.270007133483887f,
            4.059706687927246f,
            3.3751230239868164f,
            2.8007826805114746f,
            2.315382242202759f,
            1.8490285873413086f,
            1.3946564197540283f,
            0.9991465210914612f,
            0.6240804195404053f,
            0.37651097774505615f
        };

        // compare
        for (int i = 0; i < 10; i++) {
            if (Math.Abs(losses[i] - expected_losses[i]) >= 1e-2) {
                Console.Write("LOSS MISMATCH AT STEP {0}: {1} {2}\n", i, losses[i], expected_losses[i]);
                allok = false;
            } else {
                Console.Write("loss ok at step {0}: {1} {2}\n", i, losses[i], expected_losses[i]);
            }
        }

        Console.Write("overall okay: {0}\n", allok);

        // free everything
        Marshal.FreeHGlobal((IntPtr)x);
        Marshal.FreeHGlobal((IntPtr)y);
        Marshal.FreeHGlobal((IntPtr)expected_logits);
        Marshal.FreeHGlobal((IntPtr)expected_loss);
        Marshal.FreeHGlobal((IntPtr)expected_grads_memory);
        GPT2.gpt2_free(&model);

        Console.ReadKey();
    }
#endif
}