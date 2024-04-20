using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

unsafe class train_gpt2 {
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
            loader->tokens_file = stdio.fopen(filename, "rb");
            loader->file_size = stdio.fsize(loader->tokens_file);
            if (loader->file_size < (B * T + 1) * sizeof(int)) {
                throw new Exception("Error: file size is too small for the batch size and sequence length");
            }
            loader->current_position = 0; // start at the beginning

            // allocate space for B*T + 1 integers to store the inputs and targets
            loader->batch = (int*)Marshal.AllocHGlobal((B * T + 1) * sizeof(int));
            kernel32.FillMemory(loader->batch, (B * T + 1) * sizeof(int), 0);
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
            stdio.fseek(loader->tokens_file, loader->current_position, SeekOrigin.Begin);
            stdio.fread(loader->batch, sizeof(int), B * T + 1, loader->tokens_file);
            // advance the current position by B*T integers
            loader->current_position += B * T * sizeof(int);
            var current_position = stdio.fseek(loader->tokens_file, 0, SeekOrigin.Current);
            if (current_position != loader->current_position + sizeof(int)) {
                throw new IOException("Invalid file operation.");
            }
        }

        public static unsafe void dataloader_free(DataLoader* loader) {
            stdio.fclose(loader->tokens_file);
            loader->tokens_file = IntPtr.Zero;
            Marshal.FreeHGlobal((IntPtr)loader->batch);
            loader->batch = null;
        }
    }

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

    // ----------------------------------------------------------------------------
    // Tokenizer (only supports decoding)

    public unsafe struct Tokenizer {
        public int vocab_size;
        public int end_of_text;
        public byte[][] token_table;
        public bool init_ok;
        public static void tokenizer_free(ref Tokenizer tokenizer) {
            if (tokenizer.init_ok) {
                tokenizer.vocab_size = 0;
                tokenizer.token_table = null;
                tokenizer.init_ok = false;
            }
        }
        public static void tokenizer_init(ref Tokenizer tokenizer, string filename) {
            filename = Path.GetFullPath(filename);

            if (!File.Exists(filename)) {
                // try to be more helpful as we just added this feature, erase later
                Console.Write("---\n");
                Console.Write("WARNING: Failed to open the tokenizer file %s\n", filename);
                Console.Write("The Tokenizer is a new feature added April 14 2024.\n");
                Console.Write("Re-run `python train_gpt2.py` to write it\n");
                Console.Write("---\n");
                tokenizer.init_ok = false;
                return;
            }

            SafeFileHandle file = new SafeFileHandle(stdio.fopen(filename, "rb"), true);
            using (file) {
                // read in the header
                uint[] header = new uint[256];
                stdio.fread(header, file.DangerousGetHandle());
                if (header[0] != 20240328) throw new Exception("Tokenizer file is invalid.");
                if (header[1] != 1) throw new Exception("Tokenizer file is invalid.");
                tokenizer.vocab_size = (int)header[2];
                tokenizer.end_of_text = (int)header[3];
                // read in all the tokens
                // unsigned char length;
                tokenizer.token_table = new byte[tokenizer.vocab_size][];
                for (int i = 0; i < tokenizer.vocab_size; i++) {
                    byte length;
                    stdio.fread(&length, sizeof(byte), 1, file.DangerousGetHandle());
                    if (length == 0) throw new Exception("Tokenizer file is invalid.");
                    byte[] token_bytes = new byte[length + 1];
                    stdio.fread(token_bytes, length, file.DangerousGetHandle());
                    token_bytes[length] = (byte)'\0';  // Add null terminator for printing
                    tokenizer.token_table[i] = token_bytes;
                }
                tokenizer.init_ok = true;
            }
        }

        public static byte[] tokenizer_decode(ref Tokenizer tokenizer, int token_id) {
            if (!tokenizer.init_ok) {
                return null;
            }
            if (token_id < tokenizer.vocab_size) {
                return tokenizer.token_table[token_id];
            } else {
                Console.Write("invalid token id %d!\n", token_id);
                return null;
            }
        }
    }

    static void safe_printf(byte[] piece) {
        // the tokens are raw bytes, and we we only want to print the printable ones
        // many bytes can be various control codes, backspace, etc.
        if (piece == null) { return; }
        if (piece[0] == '\0') { return; }
        // handle individual byte tokens
        // every token is asserted to be at least one byte so doing piece[1] is ok
        if (piece[1] == '\0') {
            byte byte_val = (byte)piece[0];
            if (!(isprint(byte_val) || isspace(byte_val))) {
                return; // weird byte, don't print it
            }
        }
        Console.Write("{0}", Encoding.UTF8.GetString(piece));
    }

    static bool isprint(byte b) { return b >= 32 && b <= 126; }
    static bool isspace(byte b) { return b >= 9 && b <= 13; }

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
        int B = 8; // batch size 4 (i.e. 4 independent token sequences will be trained on)
        int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
        DataLoader train_loader;
        DataLoader.dataloader_init(&train_loader, train_tokens, B, T);
        Console.Write("train dataset num_batches: {0}\n", train_loader.num_batches);
        DataLoader val_loader;
        DataLoader.dataloader_init(&val_loader, val_tokens, B, T);
        Console.Write("val dataset num_batches: {0}\n", val_loader.num_batches);
        int val_num_batches = 10;

        // build the Tokenizer
        Tokenizer tokenizer = new Tokenizer();
        Tokenizer.tokenizer_init(
            ref tokenizer, "gpt2_tokenizer.bin");

        // some memory for generating samples from the model
        ulong rng_state = 1337;
        int* gen_tokens = (int*)Marshal.AllocHGlobal(B * T * sizeof(int));
        const int genT = 64; // number of steps of inference we will do

        // train
        timespec start, end;
        for (int step = 0; step <= 10000; step++) {

            // once in a while estimate the validation loss
            if (step % 100 == 0) {
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
            if (step > 0 && step % 100 == 0) {
                // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
                for (int i = 0; i < B * T; ++i) {
                    gen_tokens[i] = tokenizer.end_of_text;
                }
                // now sample from the model autoregressively
                Console.Out.Flush();
                Console.Write("generating:\n---\n");
                for (int t = 1; t < genT; t++) {
                    // note that inference is very wasteful here because for each token
                    // we re-calculate the forward pass for all of (B,T) positions from scratch
                    // but the inference here is just for sanity checking anyway
                    // and we can maybe optimize a bit more later, with careful tests
                    GPT2.gpt2_forward(&model, gen_tokens, null, B, T);
                    // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                    // we're in principle running B "inference streams" in parallel here
                    // but only using position 0
                    // get the V-dimensional vector probs[0, t-1, :]
                    float* probs = model.acts.probs + (t - 1) * model.config.vocab_size;
                    float coin = random_f32(&rng_state);
                    int next_token = sample_mult(probs, model.config.vocab_size, coin);
                    gen_tokens[t] = next_token;
                    // print the generated token, either using the Tokenizer or a fallback
                    if (tokenizer.init_ok) {
                        byte[] token_str = Tokenizer.tokenizer_decode(ref tokenizer, next_token);
                        safe_printf(token_str);
                    }
                    else
                    {
                        // fall back to printing the token id
                        Console.Write("{0} ", next_token);
                    }
                    Console.Out.Flush();
                }
                Console.Write("\n---\n");
                Console.Out.Flush();
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
        Tokenizer.tokenizer_free(ref tokenizer);
        GPT2.gpt2_free(&model);
        Marshal.FreeHGlobal((IntPtr)gen_tokens);

        Console.ReadKey();
    }
#endif
}