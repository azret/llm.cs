﻿using System;
using System.Diagnostics;

using static kernel32;
using static math;
using static time;
using static std;

#if test_gpt2_cuda
using static train_gpt2_cuda;
using static train_gpt2_cuda.GPT2;
#else 
using static train_gpt2;
using static train_gpt2.GPT2;
#endif

unsafe class test_gpt2 {
    // poor man's tensor checker
    static bool check_tensor(float* a, float* b, int n, string label) {
        int print_upto = 3;
        bool ok = true;
        float maxdiff = 0.0f;
        float tol = 2e-2f;
        printf("%s\n", label);
        for (int i = 0; i < n; i++) {
            // look at the diffence at position i of these two tensors
            float diff = fabsf(a[i] - b[i]);

            // keep track of the overall error
            ok = ok && (diff <= tol);
            if (diff > maxdiff) { maxdiff = diff; }

            // for the first few elements of each tensor, pretty print
            // the actual numbers, so we can do a visual, qualitative proof/assessment
            if (i < print_upto) {
                if (diff <= tol) {
                    if (i < print_upto) { Console.BackgroundColor = ConsoleColor.Green; printf("OK"); }
                } else {
                    if (i < print_upto) { Console.BackgroundColor = ConsoleColor.Red; printf("NOT OK"); }
                }
                Console.ResetColor();
                printf(" [%f %f]\n", a[i], b[i]);
            }
        }
        // print the final result for this tensor
        if (ok) {
            Console.BackgroundColor = ConsoleColor.Green;
            printf("TENSOR OK");
            Console.ResetColor();
            printf(", maxdiff = %e\n", maxdiff);
        } else {
            Console.BackgroundColor = ConsoleColor.Red;
            printf("TENSOR NOT OK");
            Console.ResetColor();
            printf(", maxdiff = %e\n", maxdiff);
        }
        Console.ResetColor();
        return ok;
    }

#if test_gpt2 || test_gpt2_cuda
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
        int* x = (int*) malloc(B * T * sizeof(int));
        int* y = (int*) malloc(B * T * sizeof(int));
        float* expected_logits = (float*) malloc(B * T * V * sizeof(float));
        float* expected_loss = (float*) malloc(1 * sizeof(float));
        
        // read reference information from Python
        fread(x, sizeof(int), B*T, state_file);
        fread(y, sizeof(int), B*T, state_file);
        fread(expected_logits, sizeof(float), B*T*V, state_file);
        fread(expected_loss, sizeof(float), 1, state_file);
        fread(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
        fclose(state_file);

        // overall OK signal for the test
        bool allok = true;

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
        for (int step = 0; step < 10; step++) {

            timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);

            gpt2_forward(&model, x, y, B, T);
            gpt2_zero_grad(&model);
            gpt2_backward(&model);

            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

            if (step == 0) {
                // error checking at step 0 for reference activations/gradients

                // at this point, target should be equal to expected_logits, let's compare
                int logits_ok = 1;
                for (int i=0; i<B*T*V; i++) {
                    if(i < 3) {
                        printf("%f %f\n", expected_logits[i], model.acts.logits[i]);
                    }
                    if (fabsf(expected_logits[i] - model.acts.logits[i]) >= 1e-2) {
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
                if (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                    printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                    allok = false;
                } else {
                    printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
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

            // compare the losses
            float expected_loss2 = expected_losses[step];
            float actual_loss = model.mean_loss;
            bool step_loss_ok = fabsf(expected_loss2 - actual_loss) < 1e-2;
            allok = allok && step_loss_ok;

            // print the timing information at the end
            printf("step %d: loss %f (took %f ms) OK = %d\n", step, model.mean_loss, time_elapsed_s * 1000, step_loss_ok);
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

        Console.WriteLine();
        printf("Press [Enter] to continue...");
        Console.Out.Flush();
        Console.ReadKey();
    }
#endif
}