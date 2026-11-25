//! Model config read from a Hugging Face `config.json`. Family-specific behaviour is a flag here, never a hard-coded
//! assumption in the layers, so Qwen3 0.6B/1.7B/4B and Llama-style models all load from config. Gemma needs more flags
//! (embedding scale, (1+w) norms, post-norms, GELU, SWA, per-layer inputs) and is rejected until those exist.
use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
pub struct HfConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    pub vocab_size: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
}
fn default_theta() -> f64 { 10000.0 }

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub n_q: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub theta: f32,
    pub vocab: usize,
    pub tied: bool,
    /// Qwen3: RMSNorm on q and k per head, before RoPE
    pub qk_norm: bool,
    /// Qwen2-style biases on q/k/v
    pub qkv_bias: bool,
}

impl Config {
    pub fn from_hf(h: &HfConfig) -> Result<Self, String> {
        let (qk_norm, qkv_bias) = match h.model_type.as_str() {
            "qwen3" => (true, h.attention_bias),
            "qwen2" => (false, true),
            "llama" | "mistral" => (false, h.attention_bias),
            other => return Err(format!("model_type '{other}' not supported yet (supported: qwen3, qwen2, llama, mistral)")),
        };
        if h.num_attention_heads % h.num_key_value_heads != 0 {
            return Err("num_attention_heads must be a multiple of num_key_value_heads".into());
        }
        Ok(Config {
            hidden: h.hidden_size,
            inter: h.intermediate_size,
            layers: h.num_hidden_layers,
            n_q: h.num_attention_heads,
            n_kv: h.num_key_value_heads,
            head_dim: h.head_dim.unwrap_or(h.hidden_size / h.num_attention_heads),
            eps: h.rms_norm_eps as f32,
            theta: h.rope_theta as f32,
            vocab: h.vocab_size,
            tied: h.tie_word_embeddings,
            qk_norm,
            qkv_bias,
        })
    }
}
