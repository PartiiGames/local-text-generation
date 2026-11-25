//! safetensors (bf16 / f16 / f32) -> f32 burn tensors. The gate runs f32 to match the fp32 HF reference; quantized
//! weights come with the wgpu path.
use burn::prelude::Device;
use burn::tensor::{Tensor, TensorData};
use safetensors::{Dtype, SafeTensors};

/// One or more safetensors files (a sharded checkpoint is several); a tensor is looked up across all of them.
pub struct Weights<'a> {
    st: Vec<SafeTensors<'a>>,
    dev: Device,
}

impl<'a> Weights<'a> {
    pub fn new(bytes: &'a [u8], dev: &Device) -> Result<Self, String> { Self::from_shards(&[bytes], dev) }
    pub fn from_shards(shards: &[&'a [u8]], dev: &Device) -> Result<Self, String> {
        let st = shards.iter().map(|b| SafeTensors::deserialize(b).map_err(|e| e.to_string())).collect::<Result<Vec<_>, _>>()?;
        Ok(Weights { st, dev: dev.clone() })
    }
    pub fn has(&self, name: &str) -> bool { self.st.iter().any(|s| s.tensor(name).is_ok()) }

    fn floats(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let t = self.st.iter().find_map(|s| s.tensor(name).ok()).ok_or_else(|| format!("{name}: not in any shard"))?;
        let b = t.data();
        let v: Vec<f32> = match t.dtype() {
            Dtype::BF16 => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            Dtype::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            d => return Err(format!("{name}: dtype {d:?} not supported")),
        };
        Ok((v, t.shape().to_vec()))
    }
    pub fn t1(&self, name: &str) -> Result<Tensor<1>, String> {
        let (v, s) = self.floats(name)?;
        if s.len() != 1 { return Err(format!("{name}: expected 1-D, got {s:?}")); }
        Ok(Tensor::<1>::from_data(TensorData::new(v, [s[0]]), &self.dev))
    }
    pub fn t2(&self, name: &str) -> Result<Tensor<2>, String> {
        let (v, s) = self.floats(name)?;
        if s.len() != 2 { return Err(format!("{name}: expected 2-D, got {s:?}")); }
        Ok(Tensor::<2>::from_data(TensorData::new(v, [s[0], s[1]]), &self.dev))
    }
}
