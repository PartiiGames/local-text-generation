//! Decoder-only transformer (Llama family: Qwen3, Qwen2, Llama) on burn 0.22.1 ops. Plain ops first, for the
//! correctness gate on flex. The fused wgpu kernels (RMSNorm/QK-norm, decode attention, RoPE, int8/Q4 qmm) replace pieces
//! later without changing this interface.
//!
//! Shapes: one sequence; x is [T, hidden]. The KV cache keeps k/v per layer as [n_kv, S, head_dim] (head-major, the
//! layout the shared decode-attention kernel will use).
use crate::config::{Config, HfConfig};
use crate::weights::Weights;
use burn::prelude::Device;
use burn::tensor::{activation, Bool, Int, Tensor, TensorData};

struct Linear {
    /// stored transposed: [in, out]
    wt: Tensor<2>,
    b: Option<Tensor<1>>,
}
impl Linear {
    fn load(w: &Weights, name: &str, bias: bool) -> Result<Self, String> {
        let wt = w.t2(&format!("{name}.weight"))?.transpose();
        let b = if bias { Some(w.t1(&format!("{name}.bias"))?) } else { None };
        Ok(Linear { wt, b })
    }
    fn fwd(&self, x: Tensor<2>) -> Tensor<2> {
        let y = x.matmul(self.wt.clone());
        match &self.b { Some(b) => y + b.clone().unsqueeze_dim::<2>(0), None => y }
    }
}

fn rms2(x: Tensor<2>, w: &Tensor<1>, eps: f32) -> Tensor<2> {
    let inv = (x.clone().square().mean_dim(1) + eps).sqrt().recip();
    x * inv * w.clone().unsqueeze_dim::<2>(0)
}
fn rms3(x: Tensor<3>, w: &Tensor<1>, eps: f32) -> Tensor<3> {
    let inv = (x.clone().square().mean_dim(2) + eps).sqrt().recip();
    x * inv * w.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
}

struct Layer {
    in_norm: Tensor<1>,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Option<Tensor<1>>,
    k_norm: Option<Tensor<1>>,
    post_norm: Tensor<1>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

pub struct Model {
    pub cfg: Config,
    dev: Device,
    embed: Tensor<2>,
    /// [hidden, vocab]
    lm_head_t: Tensor<2>,
    layers: Vec<Layer>,
    norm: Tensor<1>,
    inv_freq: Vec<f32>,
}

/// Per-layer K/V. Default: preallocated buffers [n_kv, cap, head_dim], written in place (slice_assign on a uniquely
/// owned tensor) and grown by doubling; attention reads the first `len` positions. `legacy` keeps the old path
/// (Tensor::cat per step, GQA by expand) for A/B runs only.
#[derive(Default)]
pub struct Cache {
    kv: Vec<Option<(Tensor<3>, Tensor<3>)>>,
    len: usize,
    cap: usize,
    #[doc(hidden)]
    pub legacy: bool,
}
impl Cache {
    /// Preallocate room for `cap` positions (e.g. persona prefix + a reply); it still grows if needed.
    pub fn with_capacity(cap: usize) -> Self { Cache { cap, ..Default::default() } }
    pub fn clear(&mut self) { self.kv.clear(); self.len = 0; }
    pub fn capacity(&self) -> usize { self.cap }
    /// Positions held (prompt + generated tokens so far).
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Drop everything after the first `n` positions, e.g. to keep a cached persona prefix and start a new turn.
    pub fn truncate(&mut self, n: usize) { self.len = self.len.min(n); }
}

impl Model {
    /// Load a Hugging Face model directory: `config.json` plus `model.safetensors` or the shards listed in
    /// `model.safetensors.index.json`. bf16 / f16 / f32 weights are converted to f32.
    pub fn from_dir(dir: impl AsRef<std::path::Path>, dev: &Device) -> Result<Self, String> {
        let dir = dir.as_ref();
        let rd = |f: &str| std::fs::read(dir.join(f)).map_err(|e| format!("{}: {e}", dir.join(f).display()));
        let hf: HfConfig = serde_json::from_slice(&rd("config.json")?).map_err(|e| format!("config.json: {e}"))?;
        let cfg = Config::from_hf(&hf)?;
        let files: Vec<String> = match rd("model.safetensors.index.json") {
            Ok(ix) => {
                let v: serde_json::Value = serde_json::from_slice(&ix).map_err(|e| format!("index: {e}"))?;
                let mut f: Vec<String> = v["weight_map"].as_object().ok_or("index: no weight_map")?.values()
                    .filter_map(|x| x.as_str().map(str::to_string)).collect();
                f.sort();
                f.dedup();
                f
            }
            Err(_) => vec!["model.safetensors".to_string()],
        };
        let shards = files.iter().map(|f| rd(f)).collect::<Result<Vec<_>, _>>()?;
        let refs: Vec<&[u8]> = shards.iter().map(|b| b.as_slice()).collect();
        Self::load(cfg, &Weights::from_shards(&refs, dev)?, dev)
    }

    /// Greedy decoding after `prompt` (appended to whatever `cache` already holds). Each new token goes to `on_token`
    /// as soon as it exists (stream it to the tokenizer / TTS); return `false` there to stop. Stops at a `stop` token
    /// (not passed on) or after `max_new` tokens. Returns the generated ids.
    pub fn generate(&self, prompt: &[u32], max_new: usize, stop: &[u32], cache: &mut Cache,
                    mut on_token: impl FnMut(u32) -> bool) -> Vec<u32> {
        let mut out = Vec::new();
        if prompt.is_empty() || max_new == 0 { return out; }
        let mut next = argmax(&self.forward(prompt, cache));
        while !stop.contains(&next) {
            out.push(next);
            if !on_token(next) || out.len() >= max_new { break; }
            next = argmax(&self.forward(&[next], cache));
        }
        out
    }

    pub fn load(cfg: Config, w: &Weights, dev: &Device) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                in_norm: w.t1(&format!("{p}.input_layernorm.weight"))?,
                q: Linear::load(w, &format!("{p}.self_attn.q_proj"), cfg.qkv_bias)?,
                k: Linear::load(w, &format!("{p}.self_attn.k_proj"), cfg.qkv_bias)?,
                v: Linear::load(w, &format!("{p}.self_attn.v_proj"), cfg.qkv_bias)?,
                o: Linear::load(w, &format!("{p}.self_attn.o_proj"), false)?,
                q_norm: if cfg.qk_norm { Some(w.t1(&format!("{p}.self_attn.q_norm.weight"))?) } else { None },
                k_norm: if cfg.qk_norm { Some(w.t1(&format!("{p}.self_attn.k_norm.weight"))?) } else { None },
                post_norm: w.t1(&format!("{p}.post_attention_layernorm.weight"))?,
                gate: Linear::load(w, &format!("{p}.mlp.gate_proj"), false)?,
                up: Linear::load(w, &format!("{p}.mlp.up_proj"), false)?,
                down: Linear::load(w, &format!("{p}.mlp.down_proj"), false)?,
            });
        }
        let embed = w.t2("model.embed_tokens.weight")?;
        let lm_head_t = if cfg.tied || !w.has("lm_head.weight") { embed.clone().transpose() } else { w.t2("lm_head.weight")?.transpose() };
        let half = cfg.head_dim / 2;
        let inv_freq = (0..half).map(|i| cfg.theta.powf(-((2 * i) as f32) / cfg.head_dim as f32)).collect();
        Ok(Model { norm: w.t1("model.norm.weight")?, embed, lm_head_t, layers, inv_freq, dev: dev.clone(), cfg })
    }

    /// cos/sin for positions pos0..pos0+t, laid out [t, 1, head_dim] (Neox half-split: the two halves repeat).
    fn rope_tables(&self, pos0: usize, t: usize) -> (Tensor<3>, Tensor<3>) {
        let d = self.cfg.head_dim;
        let (mut c, mut s) = (Vec::with_capacity(t * d), Vec::with_capacity(t * d));
        for p in pos0..pos0 + t {
            for _half in 0..2 {
                for f in &self.inv_freq {
                    let a = p as f32 * f;
                    c.push(a.cos());
                    s.push(a.sin());
                }
            }
        }
        (Tensor::<3>::from_data(TensorData::new(c, [t, 1, d]), &self.dev),
         Tensor::<3>::from_data(TensorData::new(s, [t, 1, d]), &self.dev))
    }

    fn rope(x: Tensor<3>, cos: &Tensor<3>, sin: &Tensor<3>) -> Tensor<3> {
        let [t, h, d] = x.dims();
        let x1 = x.clone().slice([0..t, 0..h, 0..d / 2]);
        let x2 = x.clone().slice([0..t, 0..h, d / 2..d]);
        let rot = Tensor::cat(vec![x2.neg(), x1], 2);
        x * cos.clone() + rot * sin.clone()
    }

    /// Runs `ids` (appended after the cache) and returns logits for the LAST position: [vocab].
    pub fn forward(&self, ids: &[u32], cache: &mut Cache) -> Vec<f32> {
        let c = &self.cfg;
        let t = ids.len();
        let pos0 = cache.len;
        let s = pos0 + t;
        if cache.kv.len() != c.layers { cache.kv = (0..c.layers).map(|_| None).collect(); }
        // Grow (or first allocate) the buffers: capacity at least s, doubling, minimum 64.
        if !cache.legacy && (s > cache.cap || cache.kv[0].is_none()) {
            let mut cap = cache.cap.max(64);
            while cap < s { cap *= 2; }
            for slot in cache.kv.iter_mut() {
                let zk = Tensor::<3>::zeros([c.n_kv, cap, c.head_dim], &self.dev);
                let zv = Tensor::<3>::zeros([c.n_kv, cap, c.head_dim], &self.dev);
                *slot = Some(match slot.take() {
                    Some((k, v)) if pos0 > 0 => (zk.slice_assign([0..c.n_kv, 0..pos0, 0..c.head_dim], k.slice([0..c.n_kv, 0..pos0, 0..c.head_dim])),
                                                 zv.slice_assign([0..c.n_kv, 0..pos0, 0..c.head_dim], v.slice([0..c.n_kv, 0..pos0, 0..c.head_dim]))),
                    _ => (zk, zv),
                });
            }
            cache.cap = cap;
        }
        let idx = Tensor::<1, Int>::from_data(TensorData::new(ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), [t]), &self.dev);
        let mut x = self.embed.clone().select(0, idx);
        let (cos, sin) = self.rope_tables(pos0, t);
        let g = c.n_q / c.n_kv;
        // causal mask: query i (absolute pos0+i) may not see key j > pos0+i. A single decode step sees every key: no mask.
        let mask_ts = (t > 1).then(|| {
            let m: Vec<bool> = (0..t).flat_map(|i| (0..s).map(move |j| j > pos0 + i)).collect();
            Tensor::<2, Bool>::from_data(TensorData::new(m, [t, s]), &self.dev)
        });
        // The grouped mask [n_kv, g*t, s] is built once per call, not per layer (per-layer expand+reshape made prefill 20 % slower).
        let mask_g = if cache.legacy { None } else {
            mask_ts.as_ref().map(|m| m.clone().unsqueeze_dim::<3>(0).expand([c.n_kv * g, t, s]).reshape([c.n_kv, g * t, s]))
        };
        let scale = (c.head_dim as f32).powf(-0.5);
        for (li, l) in self.layers.iter().enumerate() {
            let h = rms2(x.clone(), &l.in_norm, c.eps);
            let mut q = l.q.fwd(h.clone()).reshape([t, c.n_q, c.head_dim]);
            let mut k = l.k.fwd(h.clone()).reshape([t, c.n_kv, c.head_dim]);
            let v = l.v.fwd(h).reshape([t, c.n_kv, c.head_dim]);
            if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
                q = rms3(q, qn, c.eps);
                k = rms3(k, kn, c.eps);
            }
            let q = Self::rope(q, &cos, &sin).swap_dims(0, 1); // [n_q, t, d]
            let k = Self::rope(k, &cos, &sin).swap_dims(0, 1); // [n_kv, t, d]
            let v = v.swap_dims(0, 1);
            let a = if cache.legacy {
                let (k_all, v_all) = match cache.kv[li].take() {
                    Some((pk, pv)) => (Tensor::cat(vec![pk, k], 1), Tensor::cat(vec![pv, v], 1)),
                    None => (k, v),
                };
                cache.kv[li] = Some((k_all.clone(), v_all.clone()));
                // GQA: kv head j serves q heads j*g .. j*g+g-1 (HF repeat_kv order)
                let rep = |x: Tensor<3>| -> Tensor<3> {
                    if g == 1 { return x; }
                    x.unsqueeze_dim::<4>(1).expand([c.n_kv, g, s, c.head_dim]).reshape([c.n_q, s, c.head_dim])
                };
                let (kr, vr) = (rep(k_all), rep(v_all));
                let scores = q.matmul(kr.swap_dims(1, 2)) * scale; // [n_q, t, s]
                let scores = match &mask_ts { Some(m) => scores.mask_fill(m.clone().unsqueeze_dim::<3>(0).expand([c.n_q, t, s]), -1.0e9f32), None => scores };
                activation::softmax(scores, 2).matmul(vr).swap_dims(0, 1).reshape([t, c.n_q * c.head_dim])
            } else {
                let (bk, bv) = cache.kv[li].take().expect("kv buffers allocated");
                let r = [0..c.n_kv, pos0..s, 0..c.head_dim];
                let (bk, bv) = (bk.slice_assign(r.clone(), k), bv.slice_assign(r, v));
                let (k_all, v_all) = (bk.clone().slice([0..c.n_kv, 0..s, 0..c.head_dim]), bv.clone().slice([0..c.n_kv, 0..s, 0..c.head_dim]));
                cache.kv[li] = Some((bk, bv));
                // GQA without copying K/V: q heads j*g..j*g+g-1 share kv head j (HF repeat_kv order), so group them:
                // q [n_q, t, d] -> [n_kv, g*t, d]; scores [n_kv, g*t, s]
                let qg = q.reshape([c.n_kv, g * t, c.head_dim]);
                let scores = qg.matmul(k_all.swap_dims(1, 2)) * scale;
                let scores = match &mask_g { Some(m) => scores.mask_fill(m.clone(), -1.0e9f32), None => scores };
                activation::softmax(scores, 2).matmul(v_all) // [n_kv, g*t, d]
                    .reshape([c.n_q, t, c.head_dim]).swap_dims(0, 1).reshape([t, c.n_q * c.head_dim])
            };
            x = x + l.o.fwd(a);
            let h = rms2(x.clone(), &l.post_norm, c.eps);
            let m = activation::silu(l.gate.fwd(h.clone())) * l.up.fwd(h);
            x = x + l.down.fwd(m);
        }
        cache.len = s;
        let last = x.slice([t - 1..t, 0..c.hidden]);
        let logits = rms2(last, &self.norm, c.eps).matmul(self.lm_head_t.clone());
        logits.into_data().try_to_vec::<f32>().expect("logits to f32")
    }
}

pub fn argmax(v: &[f32]) -> u32 {
    let mut b = 0;
    for (i, x) in v.iter().enumerate() { if *x > v[b] { b = i; } }
    b as u32
}
