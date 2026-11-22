mod detect;

use anyhow::{Context, Result, bail};
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
};
use detect::{Detector, Output};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::Semaphore;

const INDEX_HTML: &str = include_str!("../static/index.html");

struct AppState {
    det: Detector,
    /// Only one inference at a time: the Pi Zero 2 W has 4 slow cores and 512 MB.
    gate: Semaphore,
}

#[derive(Deserialize)]
struct Params {
    conf: Option<f32>,
}

fn counts(out: &Output) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for d in &out.detections {
        *m.entry(d.label.clone()).or_insert(0) += 1;
    }
    m
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn detect_handler(
    State(st): State<Arc<AppState>>,
    Query(q): Query<Params>,
    body: Bytes,
) -> impl IntoResponse {
    let conf = q.conf.unwrap_or(0.25).clamp(0.05, 0.95);
    let _permit = st.gate.acquire().await.expect("semaphore closed");
    let st2 = st.clone();
    let res = tokio::task::spawn_blocking(move || st2.det.detect_bytes(&body, conf, 0.45)).await;
    match res {
        Ok(Ok(out)) => {
            let mut v: Value = serde_json::to_value(&out).unwrap();
            v["counts"] = json!(counts(&out));
            (StatusCode::OK, Json(v))
        }
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("{e:#}") }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    }
}

async fn info(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({ "model_input": st.det.size(), "version": env!("CARGO_PKG_VERSION") }))
}

struct Args {
    cmd: String,
    model: String,
    image: Option<String>,
    port: u16,
    size: usize,
    conf: f32,
    runs: usize,
    threads: usize,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_default();
    let mut a = Args {
        cmd,
        model: "models/yolov8n-320.onnx".into(),
        image: None,
        port: 8080,
        size: 320,
        conf: 0.35,
        runs: 3,
        threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(4),
    };
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--image" => a.image = Some(val()?),
            "--port" => a.port = val()?.parse()?,
            "--size" => a.size = val()?.parse()?,
            "--conf" => a.conf = val()?.parse()?,
            "--runs" => a.runs = val()?.parse()?,
            "--threads" => a.threads = val()?.parse()?,
            other => bail!("unknown flag {other}"),
        }
    }
    Ok(a)
}

fn usage() {
    eprintln!(
        "pi-detect — object detection with boxes and counts\n\n\
         USAGE:\n  \
         pi-detect serve --model <onnx> [--port 8080] [--size 320] [--threads 4]\n  \
         pi-detect image --model <onnx> --image <file> [--conf 0.35] [--runs 3]\n"
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    if args.threads > 1 {
        use tract_linalg::multithread::{Executor, set_default_executor};
        set_default_executor(Executor::multithread(args.threads));
    }
    eprintln!("inference threads: {}", args.threads.max(1));
    match args.cmd.as_str() {
        "serve" => {
            let det = Detector::load(&args.model, args.size)?;
            eprintln!("model loaded: {} (input {}x{})", args.model, det.size(), det.size());
            let state = Arc::new(AppState { det, gate: Semaphore::new(1) });
            let app = Router::new()
                .route("/", get(index))
                .route("/info", get(info))
                .route("/detect", post(detect_handler))
                .layer(DefaultBodyLimit::max(12 * 1024 * 1024))
                .with_state(state);
            let addr = format!("0.0.0.0:{}", args.port);
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            eprintln!("listening on http://{addr}  (open it from your phone: http://<pi-ip>:{})", args.port);
            axum::serve(listener, app).await?;
        }
        "image" => {
            let path = args.image.as_deref().context("--image is required")?;
            let bytes = std::fs::read(path)?;
            let t = std::time::Instant::now();
            let det = Detector::load(&args.model, args.size)?;
            eprintln!("model load: {:.0} ms (input {}x{})", t.elapsed().as_secs_f64() * 1e3, det.size(), det.size());
            let mut last = None;
            for i in 0..args.runs.max(1) {
                let out = det.detect_bytes(&bytes, args.conf, 0.45)?;
                eprintln!(
                    "run {}: infer {:.0} ms, total {:.0} ms",
                    i + 1,
                    out.timings.infer_ms,
                    out.timings.total_ms
                );
                last = Some(out);
            }
            let out = last.unwrap();
            for d in &out.detections {
                println!(
                    "{:<14} {:>4.0}%  box x={:.0} y={:.0} w={:.0} h={:.0}",
                    d.label, d.score * 100.0, d.x, d.y, d.w, d.h
                );
            }
            println!("counts: {:?}", counts(&out));
        }
        _ => {
            usage();
        }
    }
    Ok(())
}
