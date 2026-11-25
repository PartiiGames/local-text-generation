//! Stream a reply from a local Qwen3 model directory (Hugging Face layout), greedy, on the CPU.
//!   cargo run --release --example generate -- <model_dir> "your message"
//! Get a model first, e.g. Qwen3-0.6B (Apache-2.0, about 1.5 GB):
//!   for f in config.json tokenizer.json model.safetensors; do
//!     curl -fLO https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/$f; done
//! and check `sha256sum model.safetensors` = f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b.
use local_text_generation::{Cache, Device, Model};
use std::io::Write;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: generate <model_dir> \"message\" [max_new_tokens]");
        std::process::exit(2);
    }
    let dir = std::path::Path::new(&a[1]);
    let max_new: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(96);
    let dev = Device::flex();
    let t0 = Instant::now();
    let model = Model::from_dir(dir, &dev).unwrap_or_else(|e| { eprintln!("load: {e}"); std::process::exit(1) });
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    eprintln!("[loaded in {} ms]", t0.elapsed().as_millis());

    // Qwen3 chat template, thinking off (an empty <think> block).
    let prompt = format!("<|im_start|>system\nYou are a cheerful game buddy. Answer in one or two short sentences.<|im_end|>\n\
                          <|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", a[2]);
    let ids = tok.encode(prompt.as_str(), false).expect("encode");
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.token_to_id(t)).collect();

    let mut cache = Cache::with_capacity(ids.len() + max_new);
    let (t1, mut first, mut shown, mut got) = (Instant::now(), None, 0usize, Vec::new());
    model.generate(ids.get_ids(), max_new, &stop, &mut cache, |t| {
        first.get_or_insert_with(|| t1.elapsed());
        got.push(t);
        // Print only the new text: decode everything so far and show the part not yet printed. A token can end in
        // the middle of a multi-byte character (e.g. an emoji): the decoder then shows U+FFFD, so hold that back.
        let s = tok.decode(&got, true).unwrap_or_default();
        let s = s.trim_end_matches('\u{FFFD}');
        if s.len() > shown && s.is_char_boundary(shown) {
            print!("{}", &s[shown..]);
            std::io::stdout().flush().ok();
            shown = s.len();
        }
        true
    });
    let total = t1.elapsed();
    let ttft = first.unwrap_or(total);
    let dec = (total - ttft).as_secs_f64() / (got.len().saturating_sub(1).max(1)) as f64;
    println!();
    eprintln!("[prompt {} tokens, first token {} ms, then {:.1} tokens/s, {} tokens]", ids.len(), ttft.as_millis(), 1.0 / dec, got.len());
}
