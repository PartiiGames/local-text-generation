//! **local-text-generation**: a small, pure-Rust decoder-only LLM runtime (Qwen3 / Qwen2 / Llama-style models) on
//! [burn](https://github.com/tracel-ai/burn) 0.22, CPU first. Loads Hugging Face `config.json` + safetensors directly,
//! keeps a preallocated KV cache (reuse a persona prefix across turns), and streams tokens as they are produced.
//!
//! ```no_run
//! use local_text_generation::{Cache, Device, Model};
//! let dev = Device::flex();
//! let model = Model::from_dir("Qwen3-0.6B", &dev).unwrap();
//! let tok = tokenizers::Tokenizer::from_file("Qwen3-0.6B/tokenizer.json").unwrap();
//! let ids = tok.encode("<|im_start|>user\nSay hi!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", false).unwrap();
//! let stop = [tok.token_to_id("<|im_end|>").unwrap()];
//! let mut cache = Cache::with_capacity(512);
//! let reply = model.generate(ids.get_ids(), 64, &stop, &mut cache, |_| true);
//! println!("{}", tok.decode(&reply, true).unwrap());
//! ```
//! By Partii (https://partii.games), the place to make games together. MIT OR Apache-2.0.
pub mod config;
mod model;
mod weights;

pub use burn::prelude::Device;
pub use config::{Config, HfConfig};
pub use model::{argmax, Cache, Model};
pub use weights::Weights;
