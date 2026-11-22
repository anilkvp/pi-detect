//! Object detection core: load an ONNX YOLO model with tract, run it on an
//! image, decode boxes, apply class-wise NMS.
//!
//! Supports two common YOLO output layouts:
//!   * YOLOv8 style: [1, 4+nc, N]   (no objectness, channels first)
//!   * YOLOv5 style: [1, N, 5+nc]   (objectness + class scores)

use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;
use tract_onnx::prelude::*;

pub const COCO: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];

#[derive(Serialize, Clone, Debug)]
pub struct Detection {
    pub label: String,
    pub class_id: usize,
    pub score: f32,
    /// Top-left x/y and width/height in pixels of the image that was analysed.
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Serialize, Debug, Default, Clone)]
pub struct Timings {
    pub decode_ms: f64,
    pub preprocess_ms: f64,
    pub infer_ms: f64,
    pub postprocess_ms: f64,
    pub total_ms: f64,
}

#[derive(Serialize, Debug)]
pub struct Output {
    pub width: u32,
    pub height: u32,
    pub model_input: usize,
    pub detections: Vec<Detection>,
    pub timings: Timings,
}

type Plan = Arc<TypedRunnableModel>;

pub struct Detector {
    plan: Plan,
    size: usize,
}

impl Detector {
    /// Load the model. `size_hint` is used only if the model has a dynamic input size.
    pub fn load(path: &str, size_hint: usize) -> Result<Self> {
        let probe = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("cannot read model {path}"))?
            .into_typed()?;
        let fixed = probe
            .input_fact(0)?
            .shape
            .as_concrete()
            .filter(|s| s.len() == 4)
            .map(|s| s[2]);
        let size = fixed.unwrap_or(size_hint);

        let plan = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact([1, 3, size, size]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { plan, size })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Detect objects in encoded image bytes (JPEG/PNG).
    pub fn detect_bytes(&self, bytes: &[u8], conf: f32, iou: f32) -> Result<Output> {
        let t0 = Instant::now();
        let img = image::load_from_memory(bytes).context("could not decode image")?;
        let decode_ms = ms(t0);
        let mut out = self.detect(&img, conf, iou)?;
        out.timings.decode_ms = decode_ms;
        out.timings.total_ms += decode_ms;
        Ok(out)
    }

    pub fn detect(&self, img: &DynamicImage, conf: f32, iou: f32) -> Result<Output> {
        let s = self.size;
        let (w0, h0) = img.dimensions();

        // ---- preprocess: letterbox to s x s, gray padding, NCHW f32 in 0..1
        let t1 = Instant::now();
        let scale = (s as f32 / w0 as f32).min(s as f32 / h0 as f32);
        let nw = ((w0 as f32 * scale).round() as u32).max(1);
        let nh = ((h0 as f32 * scale).round() as u32).max(1);
        let resized = img.resize_exact(nw, nh, FilterType::Triangle).to_rgb8();
        let pad_x = (s as u32 - nw) / 2;
        let pad_y = (s as u32 - nh) / 2;
        let mut data = vec![114.0f32 / 255.0; 3 * s * s];
        for (x, y, p) in resized.enumerate_pixels() {
            let xx = (x + pad_x) as usize;
            let yy = (y + pad_y) as usize;
            for c in 0..3 {
                data[c * s * s + yy * s + xx] = p[c] as f32 / 255.0;
            }
        }
        let input = Tensor::from_shape(&[1, 3, s, s], &data)?;
        let preprocess_ms = ms(t1);

        // ---- inference
        let t2 = Instant::now();
        let result = self.plan.run(tvec!(input.into()))?;
        let infer_ms = ms(t2);

        // ---- postprocess
        let t3 = Instant::now();
        let view = result[0].to_plain_array_view::<f32>()?;
        let shape: Vec<usize> = view.shape().to_vec();
        if shape.len() != 3 {
            bail!("unexpected model output shape {shape:?}");
        }
        let flat: Vec<f32> = view.iter().copied().collect();
        let mut cands = decode(&flat, shape[1], shape[2], conf);

        // map back from letterboxed coords to original image coords
        for d in cands.iter_mut() {
            let x1 = ((d.x - pad_x as f32) / scale).clamp(0.0, w0 as f32);
            let y1 = ((d.y - pad_y as f32) / scale).clamp(0.0, h0 as f32);
            let x2 = ((d.x + d.w - pad_x as f32) / scale).clamp(0.0, w0 as f32);
            let y2 = ((d.y + d.h - pad_y as f32) / scale).clamp(0.0, h0 as f32);
            d.x = x1;
            d.y = y1;
            d.w = x2 - x1;
            d.h = y2 - y1;
        }
        let detections = nms(cands, iou);
        let postprocess_ms = ms(t3);

        let total_ms = ms(t1);
        Ok(Output {
            width: w0,
            height: h0,
            model_input: s,
            detections,
            timings: Timings {
                decode_ms: 0.0,
                preprocess_ms,
                infer_ms,
                postprocess_ms,
                total_ms,
            },
        })
    }
}

fn ms(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0
}

/// Decode raw model output to candidate boxes (still in model-input pixel space).
fn decode(flat: &[f32], d1: usize, d2: usize, conf: f32) -> Vec<Detection> {
    let mut out = Vec::new();
    // channels-first (v8): [4+nc, N]; boxes-first (v5): [N, 5+nc]
    let channels_first = d1 < d2;
    let (n, ch) = if channels_first { (d2, d1) } else { (d1, d2) };
    let get = |i: usize, c: usize| -> f32 {
        if channels_first { flat[c * n + i] } else { flat[i * ch + c] }
    };
    let has_obj = !channels_first; // v5 layout carries an objectness score
    let cls_start = if has_obj { 5 } else { 4 };
    let nc = ch - cls_start;
    for i in 0..n {
        let obj = if has_obj { get(i, 4) } else { 1.0 };
        if obj < conf {
            continue;
        }
        let (mut best, mut best_c) = (0.0f32, 0usize);
        for c in 0..nc {
            let v = get(i, cls_start + c);
            if v > best {
                best = v;
                best_c = c;
            }
        }
        let score = best * obj;
        if score < conf {
            continue;
        }
        let (cx, cy, w, h) = (get(i, 0), get(i, 1), get(i, 2), get(i, 3));
        out.push(Detection {
            label: COCO.get(best_c).copied().unwrap_or("object").to_string(),
            class_id: best_c,
            score,
            x: cx - w / 2.0,
            y: cy - h / 2.0,
            w,
            h,
        });
    }
    out
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Class-wise non-maximum suppression.
fn nms(mut cands: Vec<Detection>, thr: f32) -> Vec<Detection> {
    cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    cands.truncate(300);
    let mut keep: Vec<Detection> = Vec::new();
    for c in cands {
        if keep.iter().all(|k| k.class_id != c.class_id || iou(k, &c) < thr) {
            keep.push(c);
        }
    }
    keep
}
