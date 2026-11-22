# pi-detect — phone photo -> boxes + counts, running on a small Linux device

Rust (axum + tract-onnx, pure Rust, no Python/OpenCV at runtime). Open the device's web page on your phone,
take/choose a photo, get object boxes, labels and per-class counts. Confidence slider re-filters instantly.

## Minimum device requirements

The app needs a **Linux computer**, not a microcontroller. Raspberry Pi Pico / Pico 2 / Pico 2 W
(RP2040/RP2350, ~520 KB RAM, no OS) and Arduino/ESP32 boards **cannot run it**.

| | Minimum | Recommended |
|---|---|---|
| Board | Raspberry Pi Zero 2 W (or any 64-bit ARM / x86 Linux board) | Raspberry Pi 4 / 5, or a laptop |
| CPU | 64-bit, 4 cores (Zero 2 W: 4x Cortex-A53 @ 1 GHz) | 4+ faster cores |
| RAM | 512 MB (enough for the 320 px model) | 1 GB+ (needed for the 640 px model) |
| OS | Linux, 64-bit (Raspberry Pi OS Lite 64-bit, `aarch64`) | same |
| Storage | microSD 8 GB+ (16 GB+ comfortable); the OS image alone needs ~3.1 GB | 16 GB+ |
| Network | Wi-Fi or Ethernet on the same network as your phone (Zero 2 W: **2.4 GHz only**) | same |
| Power | 5 V via the port marked **PWR IN** (phone charger, 1 A+; 2 A is safer) | official PSU |
| Camera / screen / keyboard | Not needed. The phone supplies the photo; set up the Pi headless over SSH | |

Other requirements:
- **Phone:** any modern browser (Chrome/Safari). Phone and device must be on the same network. VPNs can block local access, so turn them off.
- **Laptop (for building and setup):** Linux/macOS/Windows with Rust (`rustup`), and a microSD card slot or USB reader.
- **32-bit OS:** possible (target `armv7-unknown-linux-gnueabihf`) but untested; use 64-bit.

Approximate model needs (check against your device):

| Model | File size | RAM while running | Speed (x86 sandbox, 1 image) | Use when |
|---|---|---|---|---|
| `yolov8n-320.onnx` | ~12 MB | ~100-150 MB | ~30-50 ms | Zero 2 W (default) |
| `yolov8n-640.onnx` | ~12 MB | ~250-400 MB | ~400 ms | Pi 4/5 or laptop, small objects |

Pi Zero 2 W speed is **not measured yet**; expect roughly 0.3-1 s per photo at 320 px.

## 1. Build (pick one)
Check the device OS first: `uname -m` -> `aarch64` (64-bit) or `armv7l` (32-bit).

**A. Cross-compile on your laptop (fastest)**

    cargo install cargo-zigbuild && pip install ziglang
    rustup target add aarch64-unknown-linux-gnu      # armv7-unknown-linux-gnueabihf for 32-bit OS
    cargo zigbuild --release --target aarch64-unknown-linux-gnu

Binary: `target/aarch64-unknown-linux-gnu/release/pi-detect`

(Alternative to Zig: `aarch64-linux-gnu-gcc` plus a `.cargo/config.toml` linker entry. Zig lets you target an older glibc, e.g. `--target aarch64-unknown-linux-gnu.2.31`, which avoids "GLIBC not found" errors on the Pi.)

**B. Build on the Pi** (slow, 30+ min, needs RAM headroom): add 1 GB swap first, then `cargo build --release -j1`.

**C. Run on your laptop (no Pi needed)**

    cargo run --release -- serve --model models/yolov8n-320.onnx --port 8080

## 2. Copy to the Pi

    scp pi-detect models/yolov8n-320.onnx <user>@<pi-ip-or-hostname>:~/

(The web page is embedded in the binary.)

## 3. Run

    chmod +x pi-detect
    ./pi-detect serve --model yolov8n-320.onnx --port 8080

Open `http://<pi-ip>:8080` on a phone on the same Wi-Fi. Tap the photo button.
Flags: `--size 320` (must match the model), `--conf 0.35`, `--threads 4`.
Test without the phone: `./pi-detect image --model yolov8n-320.onnx --image test.jpg --runs 3`

Re-export a model yourself (needs Python on a laptop, not on the Pi):

    pip install ultralytics && yolo export model=yolov8n.pt format=onnx imgsz=320 opset=12

FP16 ONNX files will not load; export FP32.

## 4. Autostart (optional) — /etc/systemd/system/pi-detect.service

    [Unit]
    Description=pi-detect
    After=network-online.target
    [Service]
    User=<user>
    WorkingDirectory=/home/<user>
    ExecStart=/home/<user>/pi-detect serve --model yolov8n-320.onnx --port 8080
    Restart=always
    [Install]
    WantedBy=multi-user.target

Then `sudo systemctl enable --now pi-detect`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `cannot execute binary file` | OS is 32-bit, or the wrong target was built. Use 64-bit OS + `aarch64` build. |
| `GLIBC_x.y not found` | Build with Zig and an older glibc target (see Build A). |
| Phone cannot open the page | Same Wi-Fi as the device? VPN off? Firewall allows port 8080? |
| Out of memory / killed | Use the 320 px model, close other services, add swap. |
| Model fails to load | FP16 model; re-export as FP32. |

## Notes
- Phone camera capture over plain http works via file upload (the `capture` input); live getUserMedia would need https.
- Ultralytics YOLO weights are AGPL-3.0. Fine for a demo; for a product use an Apache/MIT-licensed detector
  or buy an Ultralytics licence.
- Tested on x86 Linux only. Not yet run on a real Pi.
