//! chat-gate: checks the burn decoder against a Hugging Face reference (tools/hf_ref.py, fp32, greedy).
//!   chat-gate <model_dir> <ref.json>
//! Per case: (1) our tokenizer reproduces HF's prompt ids from the templated text; (2) teacher-forced, so one early
//! miss can't cascade: at every step our argmax must equal HF's greedy token; (3) logit max-abs-diff on HF's top-50
//! at step 0 and on the chosen token at every step. A miss where HF's own top-2 margin is < 0.01 is a near-tie and is
//! reported apart. Prints one GATE line per case, then GATE SUMMARY {json} and GATE PASS / GATE FAIL.
use burn::prelude::Device;
use local_text_generation::{argmax, Cache, Config, HfConfig, Model};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct Case {
    text: String,
    prompt_ids: Vec<u32>,
    gen_ids: Vec<u32>,
    gen_logit: Vec<f32>,
    gen_margin: Vec<f32>,
    top_ids: Vec<u32>,
    top_logits: Vec<f32>,
}
#[derive(Deserialize)]
struct Ref { model: String, cases: Vec<Case> }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: chat-gate <model_dir> <ref.json>"); std::process::exit(2); }
    let dir = std::path::Path::new(&a[1]);
    let rf: Ref = serde_json::from_slice(&std::fs::read(&a[2]).expect("read ref")).expect("parse ref");
    let hf: HfConfig = serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config")).expect("parse config");
    let cfg = Config::from_hf(&hf).unwrap_or_else(|e| { eprintln!("config: {e}"); std::process::exit(2) });
    println!("GATE start model={} layers={} hidden={} heads={}/{} head_dim={} cases={} threads={}", rf.model, cfg.layers,
        cfg.hidden, cfg.n_q, cfg.n_kv, cfg.head_dim, rf.cases.len(), std::env::var("RAYON_NUM_THREADS").unwrap_or_default());
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let dev = Device::flex();
    let t0 = Instant::now();
    let model = Model::from_dir(dir, &dev).expect("load");
    println!("GATE loaded in {} ms", t0.elapsed().as_millis());

    let (mut steps, mut hits, mut ties, mut tok_ok, mut first_div_sum) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut max_top, mut max_chosen) = (0f32, 0f32);
    let (mut prefill_ms, mut prefill_tok, mut dec_ms, mut dec_tok) = (0f64, 0usize, 0f64, 0usize);
    for (ci, c) in rf.cases.iter().enumerate() {
        let enc = tok.encode(c.text.as_str(), false).expect("encode");
        let ours: Vec<u32> = enc.get_ids().to_vec();
        let tk = ours == c.prompt_ids;
        if tk { tok_ok += 1; }
        let mut cache = Cache::default();
        let t = Instant::now();
        let mut logits = model.forward(&c.prompt_ids, &mut cache);
        prefill_ms += t.elapsed().as_secs_f64() * 1e3; prefill_tok += c.prompt_ids.len();
        let top = c.top_ids.iter().zip(&c.top_logits).map(|(&i, &l)| (logits[i as usize] - l).abs()).fold(0f32, f32::max);
        max_top = max_top.max(top);
        let (mut h, mut tie, mut first_div, mut chosen) = (0, 0, c.gen_ids.len(), 0f32);
        for (s, &g) in c.gen_ids.iter().enumerate() {
            let ok = argmax(&logits) == g;
            if ok { h += 1; } else {
                if c.gen_margin[s] < 0.01 { tie += 1; }
                if first_div == c.gen_ids.len() { first_div = s; }
            }
            chosen = chosen.max((logits[g as usize] - c.gen_logit[s]).abs());
            if s + 1 < c.gen_ids.len() {
                let t = Instant::now();
                logits = model.forward(&[g], &mut cache);
                dec_ms += t.elapsed().as_secs_f64() * 1e3; dec_tok += 1;
            }
        }
        max_chosen = max_chosen.max(chosen);
        steps += c.gen_ids.len(); hits += h; ties += tie; first_div_sum += first_div;
        let text = tok.decode(&c.gen_ids, true).unwrap_or_default().replace('\n', " ");
        println!("GATE case {ci}: tok {} ({}/{} ids), top1 {h}/{} (near-ties {tie}), free-run match {first_div} steps, top50 diff {top:.4}, chosen diff {chosen:.4} | {}",
            if tk { "ok" } else { "MISMATCH" }, ours.len(), c.prompt_ids.len(), c.gen_ids.len(), text.chars().take(80).collect::<String>());
    }
    // Long context, the buddy's real shape (persona + state prefix, then a short reply): prefill ~LONG tokens, decode 32.
    // New KV path (preallocated, grouped GQA) vs legacy (cat + expand), interleaved ABAB for 3 rounds in this process.
    // Both must pick the same greedy tokens.
    let long = std::env::var("GATE_LONG").ok().and_then(|v| v.parse().ok()).unwrap_or(512usize);
    let mut ctx: Vec<u32> = Vec::new();
    while ctx.len() < long { for c in &rf.cases { ctx.extend(&c.prompt_ids); } }
    ctx.truncate(long);
    let (mut lc_new, mut lc_old, mut lc_same, mut lc_pre_new, mut lc_pre_old) = (Vec::new(), Vec::new(), true, Vec::new(), Vec::new());
    let mut first_toks: Option<Vec<u32>> = None;
    for round in 0..3 {
        for legacy in [false, true] {
            let mut cache = Cache::default();
            cache.legacy = legacy;
            let t = Instant::now();
            let mut logits = model.forward(&ctx, &mut cache);
            let pre = t.elapsed().as_secs_f64() * 1e3;
            let mut toks = Vec::new();
            let t = Instant::now();
            for _ in 0..32 {
                let nx = argmax(&logits);
                toks.push(nx);
                logits = model.forward(&[nx], &mut cache);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / 32.0;
            match &first_toks { None => first_toks = Some(toks.clone()), Some(f) => if *f != toks { lc_same = false; } }
            println!("GATE longctx round {round} {}: prefill {long} tok {pre:.0} ms, decode {ms:.1} ms/tok at S~{}",
                if legacy { "legacy" } else { "new" }, long + 16);
            if legacy { lc_old.push(ms); lc_pre_old.push(pre); } else { lc_new.push(ms); lc_pre_new.push(pre); }
        }
    }
    let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.total_cmp(b)); v[v.len() / 2] };
    let (dn, dl) = (med(lc_new), med(lc_old));
    println!("GATE longctx median decode: new {dn:.1} ms/tok, legacy {dl:.1} ms/tok ({:.2}x); prefill new {:.0} ms, legacy {:.0} ms; same tokens: {lc_same}",
        dl / dn, med(lc_pre_new), med(lc_pre_old));

    let n = rf.cases.len();
    let rate = hits as f64 / steps.max(1) as f64;
    let rate_ex = (hits + ties) as f64 / steps.max(1) as f64;
    let pass = tok_ok == n && rate >= 0.99 && max_top < 0.1 && lc_same;
    println!("GATE SUMMARY {}", serde_json::json!({
        "model": rf.model, "cases": n, "tokenizer_ok": tok_ok, "steps": steps, "top1": hits, "top1_rate": rate,
        "near_ties": ties, "top1_rate_excl_ties": rate_ex, "mean_free_run_steps": first_div_sum as f64 / n.max(1) as f64,
        "max_top50_logit_diff": max_top, "max_chosen_logit_diff": max_chosen,
        "prefill_tok_s": prefill_tok as f64 / (prefill_ms / 1e3).max(1e-9), "decode_tok_s": dec_tok as f64 / (dec_ms / 1e3).max(1e-9),
        "decode_ms_per_tok": dec_ms / dec_tok.max(1) as f64, "longctx": {"prefill": long, "decode_ms_new": dn, "decode_ms_legacy": dl, "same_tokens": lc_same},
        "gate": "tokenizer all, top1 >= 0.99, top50 diff < 0.1, long-context new == legacy tokens" }));
    println!("{}", if pass { "GATE PASS" } else { "GATE FAIL" });
    std::process::exit(if pass { 0 } else { 1 });
}
