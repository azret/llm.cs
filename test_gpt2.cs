using System;
using System.Diagnostics;

using static kernel32;
using static GPT2;
using static math;
using static time;
using static std;

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

#if test_gpt2
    static unsafe void Main(string[] args) {
        // build the GPT-2 model from a checkpoint
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        int C = model.config.channels;
        int V = model.config.vocab_size;
        int maxT = model.config.max_seq_len;
        int L = model.config.num_layers;

        // load additional information that we will use for debugging and error checking
        IntPtr state_file = fopen("gpt2_124M_debug_state.bin", "rb");
        int[] state_header = new int[256];
        fread(state_header, state_file);
        if (state_header[0] != 20240327) { throw new Exception("Bad magic state file"); }
        if (state_header[1] != 1) { throw new Exception("Bad version in state file"); }
        int B = state_header[2]; // batch size, e.g. 4
        int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
        printf("[State]\n");
        printf("batch_size: %d\n", B);
        printf("seq_len: %d\n", T);

        ParameterTensors expected_grads;
        float* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes);

        // inputs and expected outputs, only used for error checking
        int* x = (int*)malloc(B * T * sizeof(int));
        int* y = (int*)malloc(B * T * sizeof(int));
        float* expected_logits = (float*)malloc(B * T * V * sizeof(float));
        float* expected_loss = (float*)malloc(1 * sizeof(float));
        
        // read reference information from Python
        fread(x, sizeof(int), B*T, state_file);
        fread(y, sizeof(int), B*T, state_file);
        fread(expected_logits, sizeof(float), B*T*V, state_file);
        fread(expected_loss, sizeof(float), 1, state_file);
        fread(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
        fclose(state_file);

        // overall OK signal for the test
        bool allok = true;

        // let's do 10 training iterations, following the pytorch code
        float[] losses = new float[10];
        for (int step = 0; step < 10; step++) {

            timespec start, end;
            getTicks(CLOCK_MONOTONIC, &start);

            gpt2_forward(&model, x, y, B, T);
            gpt2_zero_grad(&model);
            gpt2_backward(&model);

            getTicks(CLOCK_MONOTONIC, &end);

            if (step == 0) {
                // error checking at step 0 for reference activations/gradients

                // at this point, target should be equal to expected_logits, let's compare
                int logits_ok = 1;
                for (int i=0; i<B*T*V; i++) {
                    if(i < 3) {
                        printf("%f %f\n", expected_logits[i], model.acts.logits[i]);
                    }
                    if (Math.Abs(expected_logits[i] - model.acts.logits[i]) >= 1e-2) {
                        printf("MISMATCH AT INDEX %d: ", i);
                        printf("%f %f\n", expected_logits[i],model.acts.logits[i]);
                        logits_ok = 0;
                        break;
                    }
                }
                if(logits_ok == 0) { printf("NOT "); }

                printf("OK (LOGITS)\n");
                allok = allok && logits_ok == 1;

                // compare the achieved loss
                if (Math.Abs(model.mean_loss - *expected_loss) >= 1e-2) {
                    Console.Write("LOSS MISMATCH: {0} {1}\n", model.mean_loss, *expected_loss);
                    allok = false;
                } else {
                    Console.Write("LOSS OK: {0} {1}\n", model.mean_loss, *expected_loss);
                }

                // finally check all the gradients
                bool[] gradoks =  new bool[16];
                ParameterTensors grads = model.grads;
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

            gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

            // print the timing information at the end
            double time_elapsed_ms = (end.secs - start.secs);
            Console.Write("step {0}: loss {1} (took {2} ms)\n", step, model.mean_loss, time_elapsed_ms * 1000);
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
        if (allok) {
            Console.BackgroundColor = ConsoleColor.Green;
        } else {
            Console.BackgroundColor = ConsoleColor.Red;
        }
        Console.Write("overall okay: {0}", allok);
        Console.ResetColor();


        // free everything
        free(x);
        free(y);
        free(expected_logits);
        free(expected_loss);
        free(expected_grads_memory);
        gpt2_free(&model);

        Console.ReadKey();
    }
#endif
}