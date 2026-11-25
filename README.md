# local-text-generation

On-device LLM chat in pure Rust, by Partii (https://partii.games), the place to make games together.

A small runtime for decoder-only LLMs (Qwen3, Qwen2 and Llama-style models) on
[burn](https://github.com/tracel-ai/burn) 0.22. It loads a Hugging Face model directory directly (`config.json` +
safetensors, bf16/f16/f32), keeps a preallocated KV cache you can reuse across turns, and streams each token as soon as
it exists. No C or C++ dependencies, no Python at runtime.

It is the chat engine behind the talking AI buddy in Partii games, where a small model on
the player's own device writes short in-character replies.

## Example

```sh
mkdir Qwen3-0.6B && cd Qwen3-0.6B
for f in config.json tokenizer.json model.safetensors; do
  curl -fLO https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/$f
done
sha256sum model.safetensors   # f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
cd .. && cargo run --release --example generate -- Qwen3-0.6B "Cover me, I'm going in!"
```

In code:

```rust,no_run
use local_text_generation::{Cache, Device, Model};

let dev = Device::flex();
let model = Model::from_dir("Qwen3-0.6B", &dev)?;
let tok = tokenizers::Tokenizer::from_file("Qwen3-0.6B/tokenizer.json")?;
let prompt = "<|im_start|>user\nSay hi!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
let ids = tok.encode(prompt, false)?;
let stop = [tok.token_to_id("<|im_end|>").unwrap()];

let mut cache = Cache::with_capacity(512);
let reply = model.generate(ids.get_ids(), 64, &stop, &mut cache, |_token| true /* keep going */);
println!("{}", tok.decode(&reply, true)?);
```

- `Model::forward(ids, &mut cache)` runs any number of new tokens after what the cache holds and returns the last
  position's logits, if you want your own sampling.
- `Cache::truncate(n)` keeps the first `n` positions, e.g. a cached persona prompt, and drops the rest before the next
  turn.

## Status
- **CPU (burn's flex backend), f32.** Checked against Hugging Face transformers (fp32, eager) on Qwen3-0.6B. Over 20
  prompts, greedy top-1 matched on 288/288 teacher-forced steps, with max logit difference 5e-5. Long-context new and
  reference cache paths produce identical tokens.
- **Speed (Qwen3-0.6B, f32, 4 threads on an x86 server):** prefill about 260 tokens/s, decode about 16 tokens/s.
  Decode at this size is bound by reading the f32 weights, so int8/Q4 weights are the next big win.
- **Planned:** int8 and Q4 weights; a GPU path (wgpu, Metal); grammar-constrained JSON output.
- **Not supported yet:** Gemma-family models (different norms, embeddings scale, sliding window). `Config::from_hf`
  refuses them with a clear error rather than running them wrong.

## Licence
MIT OR Apache-2.0, at your option. The crate contains no model weights. The models you load have their own licences;
Qwen3 is Apache-2.0. See [NOTICE](NOTICE) for the tested weight files (revision and sha256) and dependency licences.
