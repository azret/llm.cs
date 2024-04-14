using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

unsafe class train_gpt2 {
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

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct DataLoader {
        // hyperparameters
        public int B; // batch size
        public int T; // sequence length
        // input handling and its state
        public IntPtr tokens_file;
        public long file_size;
        public long current_position;
        // output memory
        public int* batch;
        public int* inputs;
        public int* targets;
        // convenience variables
        public int num_batches;

        public static unsafe void dataloader_init(DataLoader* loader, string filename, int B, int T) {
            loader->B = B;
            loader->T = T;

            // open the input file for reading
            loader->tokens_file = Win32.fopen(filename, "rb");
            loader->file_size = Win32.fsize(loader->tokens_file);
            if (loader->file_size < (B * T + 1) * sizeof(int)) {
                throw new Exception("Error: file size is too small for the batch size and sequence length");
            }
            loader->current_position = 0; // start at the beginning

            // allocate space for B*T + 1 integers to store the inputs and targets
            loader->batch = (int*)Marshal.AllocHGlobal((B * T + 1) * sizeof(int));
            Win32.FillMemory(loader->batch, (B * T + 1) * sizeof(int), 0);
            loader->inputs = loader->batch;
            loader->targets = loader->batch + 1; // targets are shifted by one
            loader->num_batches = (int)((long)loader->file_size / (B * T * sizeof(int)));
        }

        public static unsafe void dataloader_reset(DataLoader* loader) {
            loader->current_position = 0; // start at the beginning
        }

        public static unsafe void dataloader_next_batch(DataLoader* loader) {
            int B = loader->B;
            int T = loader->T;
            // if we are at the end of the file, loop back to the beginning
            if (loader->current_position + (B * T + 1) * sizeof(int) > loader->file_size) {
                loader->current_position = 0;
            }
            // read the B*T+1 integers from the file into batch
            Win32.fseek(loader->tokens_file, loader->current_position, SeekOrigin.Begin);
            Win32.fread(loader->batch, sizeof(int), B * T + 1, loader->tokens_file);
            // advance the current position by B*T integers
            loader->current_position += B * T * sizeof(int);
            var current_position = Win32.fseek(loader->tokens_file, 0, SeekOrigin.Current);
            if (current_position != loader->current_position + sizeof(int)) {
                throw new IOException("Invalid file operation.");
            }
        }

        public static unsafe void dataloader_free(DataLoader* loader) {
            Win32.fclose(loader->tokens_file);
            loader->tokens_file = IntPtr.Zero;
            Marshal.FreeHGlobal((IntPtr)loader->batch);
            loader->batch = null;
        }
    }

    // the GPT-2 end-of-text token id
    const int GPT2_EOT = 50256;

    static unsafe ulong random_u32(ulong* state) {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        return (*state * 0x2545F4914F6CDD1Dul) >> 32;
    }
    static unsafe float random_f32(ulong* state) { // random float32 in [0,1)
        return (random_u32(state) >> 8) / 16777216.0f;
    }

    static unsafe int sample_mult(float* probabilities, int n, float coin) {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

#if train_gpt2
    static unsafe void Main(string[] args) {
        // build the GPT-2 model from a checkpoint
        GPT2 model;
        GPT2.gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
        string tiny_stories_train = "data/TinyStories_train.bin";
        string tiny_stories_val = "data/TinyStories_val.bin";
        string tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
        string tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";
        string train_tokens = File.Exists(tiny_shakespeare_train) ? tiny_shakespeare_train : tiny_stories_train;
        string val_tokens = File.Exists(tiny_shakespeare_val) ? tiny_shakespeare_val : tiny_stories_val;
        int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
        int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
        DataLoader train_loader;
        DataLoader.dataloader_init(&train_loader, train_tokens, B, T);
        Console.Write("train dataset num_batches: {0}\n", train_loader.num_batches);
        DataLoader val_loader;
        DataLoader.dataloader_init(&val_loader, val_tokens, B, T);
        Console.Write("val dataset num_batches: {0}\n", val_loader.num_batches);
        int val_num_batches = 10;

        // some memory for generating samples from the model
        ulong rng_state = 1337;
        const int gen_max_length = 64;
        // during inference step we'll generate sequences of this many tokens
        int* gen_tokens = (int*)Marshal.AllocHGlobal(gen_max_length * sizeof(int));

        // train
        timespec start, end;
        for (int step = 0; step <= 20; step++) {

            // once in a while estimate the validation loss
            if (step % 10 == 0) {
                float val_loss = 0.0f;
                DataLoader.dataloader_reset(&val_loader);
                for (int i = 0; i < val_num_batches; i++) {
                    DataLoader.dataloader_next_batch(&val_loader);
                    GPT2.gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                    val_loss += model.mean_loss;
                }
                val_loss /= val_num_batches;
                Console.Write("val loss {0}\n", val_loss);
            }

            // once in a while do model inference to print generated text
            if (step > 0 && step % 20 == 0) {
                gen_tokens[0] = GPT2_EOT; // the GPT-2 EOT token kicks off the generation
                for (int t = 1; t < gen_max_length; t++) {
                    // note that inference is wasteful here because
                    // for each t, we re-compute all activations between 0 and t
                    // leaving this alone because you want separate code for inference anyway
                    // the inference here is just for sanity checking purposes
                    GPT2.gpt2_forward(&model, gen_tokens, null, 1, t);
                    float* probs = model.acts.probs + (t-1) * model.config.vocab_size;
                    float coin = random_f32(&rng_state);
                    int next_token = sample_mult(probs, model.config.vocab_size, coin);
                    gen_tokens[t] = next_token;
                }
                Console.Write("generated: ");
                for (int t = 0; t < gen_max_length; t++) {
                    Console.Write("{0} ", gen_tokens[t]);
                }
                Console.Write("\n");
            }

            // do a training step
            clock_gettime(CLOCK_MONOTONIC, &start);
            DataLoader.dataloader_next_batch(&train_loader);
            GPT2.gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
            GPT2.gpt2_zero_grad(&model);
            GPT2.gpt2_backward(&model);
            GPT2.gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

            Console.Write("step {0}: train loss {1} (took {2} ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
        }

        // free
        DataLoader.dataloader_free(&train_loader);
        DataLoader.dataloader_free(&val_loader);
        GPT2.gpt2_free(&model);

        Console.ReadKey();
    }
#endif
}