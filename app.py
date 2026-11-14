"""AeroVision AI - Streamlit dashboard.

Upload photos and videos, run the trained YOLO model on them and show the
results on an airport-style "departures board".

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
from html import escape
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "best.pt"
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
DISPLAY_WIDTH = 960  # frames are resized to this width before display

# --------------------------------------------------------------------------
# Visual theme (dark control-tower look, amber flight-board digits)
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');

:root {
    --bg: #070B14;
    --panel: #0F1729;
    --line: #1E2A44;
    --amber: #FFB400;
    --cyan: #38BDF8;
    --green: #34D399;
    --text: #E6EDF7;
    --muted: #8393B2;
}

.stApp {
    background: radial-gradient(1200px 600px at 15% -10%, #14254A 0%, #070B14 60%);
    color: var(--text);
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

section[data-testid="stSidebar"] {
    background: #0A1120;
    border-right: 1px solid var(--line);
}

.av-header { padding: 6px 0 2px 0; }
.av-title {
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-weight: 700;
    font-size: 2.3rem;
    letter-spacing: 0.18em;
    color: var(--amber);
    text-shadow: 0 0 18px rgba(255, 180, 0, 0.35);
}
.av-subtitle {
    color: var(--muted);
    letter-spacing: 0.22em;
    font-size: 0.8rem;
    text-transform: uppercase;
}
.av-runway {
    height: 6px;
    margin: 14px 0 22px 0;
    border-radius: 3px;
    background: repeating-linear-gradient(90deg, var(--cyan) 0 28px, transparent 28px 52px);
    opacity: 0.55;
}

.av-cards { display: flex; gap: 14px; margin: 6px 0 18px 0; flex-wrap: wrap; }
.av-card {
    flex: 1 1 150px;
    background: linear-gradient(180deg, #111C34 0%, var(--panel) 100%);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 18px;
}
.av-card .label {
    color: var(--muted);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.av-card .value {
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-weight: 700;
    font-size: 1.9rem;
    color: var(--cyan);
}

.av-board {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 6px;
    font-family: 'JetBrains Mono', Consolas, monospace;
}
.av-board th {
    color: var(--muted);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-align: left;
    padding: 4px 14px;
    text-transform: uppercase;
}
.av-board td {
    background: #0B1326;
    color: var(--amber);
    font-size: 1.05rem;
    font-weight: 700;
    padding: 10px 14px;
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    text-transform: uppercase;
}
.av-board td:first-child { border-left: 1px solid var(--line); border-radius: 8px 0 0 8px; }
.av-board td:last-child { border-right: 1px solid var(--line); border-radius: 0 8px 8px 0; }
.av-board td.status { color: var(--green); }
.av-empty {
    color: var(--muted);
    font-family: 'JetBrains Mono', Consolas, monospace;
    padding: 18px;
    border: 1px dashed var(--line);
    border-radius: 10px;
    text-align: center;
    letter-spacing: 0.15em;
}

.av-file {
    font-family: 'JetBrains Mono', Consolas, monospace;
    color: var(--text);
    font-size: 1.05rem;
    letter-spacing: 0.08em;
    margin: 22px 0 8px 0;
    padding-left: 12px;
    border-left: 4px solid var(--amber);
}

div.stButton > button {
    background: var(--amber);
    color: #0B1326;
    font-weight: 800;
    letter-spacing: 0.15em;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
}
div.stButton > button:hover { background: #FFC93C; color: #0B1326; }
</style>
"""


# --------------------------------------------------------------------------
# Model and inference helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(model_path: str) -> YOLO:
    """Load the YOLO weights once and keep them in memory between reruns."""
    return YOLO(model_path)


def run_detection(model: YOLO, frame_bgr: np.ndarray, conf: float):
    """Run the model on one BGR frame.

    Returns the annotated frame (BGR) and a list of (class_name, confidence).
    """
    result = model.predict(frame_bgr, conf=conf, verbose=False)[0]
    annotated = result.plot()  # draws boxes and labels, returns BGR
    detections = [
        (result.names[int(box.cls)], float(box.conf)) for box in result.boxes
    ]
    return annotated, detections


def to_display(frame_bgr: np.ndarray) -> np.ndarray:
    """Resize (if too wide) and convert BGR to RGB for Streamlit."""
    height, width = frame_bgr.shape[:2]
    if width > DISPLAY_WIDTH:
        scale = DISPLAY_WIDTH / width
        frame_bgr = cv2.resize(frame_bgr, (DISPLAY_WIDTH, int(height * scale)))
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def update_stats(stats: dict, detections: list[tuple[str, float]]) -> None:
    """Accumulate per-class statistics from the detections of one frame."""
    per_frame = Counter(name for name, _ in detections)
    for name, count in per_frame.items():
        entry = stats.setdefault(name, {"max": 0, "total": 0, "conf": 0.0})
        entry["max"] = max(entry["max"], count)
    for name, confidence in detections:
        stats[name]["total"] += 1
        stats[name]["conf"] += confidence


def render_cards(items: list[tuple[str, str]]) -> str:
    cards = "".join(
        f'<div class="av-card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div></div>'
        for label, value in items
    )
    return f'<div class="av-cards">{cards}</div>'


def render_board(stats: dict) -> str:
    """Build the flight-board style table with one row per detected class."""
    if not stats:
        return '<div class="av-empty">SIN DETECCIONES</div>'
    rows = ""
    for name, entry in sorted(stats.items(), key=lambda kv: -kv[1]["total"]):
        mean_conf = entry["conf"] / entry["total"]
        rows += (
            f"<tr><td>{escape(name)}</td><td>{entry['max']:02d}</td>"
            f"<td>{entry['total']:04d}</td><td>{mean_conf:.0%}</td>"
            '<td class="status">DETECTADO</td></tr>'
        )
    return (
        '<table class="av-board"><thead><tr><th>Objeto</th>'
        "<th>Max. en escena</th><th>Detecciones</th><th>Conf. media</th>"
        f"<th>Estado</th></tr></thead><tbody>{rows}</tbody></table>"
    )


# --------------------------------------------------------------------------
# File processing
# --------------------------------------------------------------------------
def process_image(model: YOLO, uploaded, conf: float) -> None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("No se pudo leer la imagen.")
        return

    annotated, detections = run_detection(model, frame, conf)
    stats: dict = {}
    update_stats(stats, detections)

    left, right = st.columns([3, 2])
    with left:
        st.image(to_display(annotated))
    with right:
        st.markdown(
            render_cards([("Objetos detectados", str(len(detections)))]),
            unsafe_allow_html=True,
        )
        st.markdown(render_board(stats), unsafe_allow_html=True)


def process_video(model: YOLO, uploaded, conf: float, stride: int) -> None:
    suffix = Path(uploaded.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()  # closed so OpenCV can open it on Windows

    capture = cv2.VideoCapture(tmp.name)
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        left, right = st.columns([3, 2])
        frame_slot = left.empty()
        cards_slot = right.empty()
        board_slot = right.empty()
        progress = st.progress(0.0)

        stats: dict = {}
        read_count = 0
        processed = 0
        started = time.time()

        while True:
            ok = capture.grab()  # cheap: advances without decoding
            if not ok:
                break
            read_count += 1
            if (read_count - 1) % stride != 0:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                break

            annotated, detections = run_detection(model, frame, conf)
            update_stats(stats, detections)
            processed += 1
            frame_slot.image(to_display(annotated))

            if processed % 5 == 0:
                elapsed = max(time.time() - started, 1e-6)
                cards_slot.markdown(
                    render_cards(
                        [
                            ("Fotogramas", f"{processed}"),
                            ("En escena", str(len(detections))),
                            ("FPS", f"{processed / elapsed:.1f}"),
                        ]
                    ),
                    unsafe_allow_html=True,
                )
                board_slot.markdown(render_board(stats), unsafe_allow_html=True)
                progress.progress(min(read_count / total_frames, 1.0))

        progress.progress(1.0)
        cards_slot.markdown(
            render_cards(
                [
                    ("Fotogramas analizados", f"{processed}"),
                    ("Clases detectadas", str(len(stats))),
                    ("Estado", "COMPLETADO"),
                ]
            ),
            unsafe_allow_html=True,
        )
        board_slot.markdown(render_board(stats), unsafe_allow_html=True)
    finally:
        capture.release()
        os.unlink(tmp.name)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="AeroVision AI", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="av-header"><div class="av-title">AEROVISION AI</div>'
        '<div class="av-subtitle">Centro de control de deteccion aerea</div></div>'
        '<div class="av-runway"></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Panel de control")
        model_path = st.text_input("Ruta del modelo", str(DEFAULT_MODEL))
        conf = st.slider("Confianza minima", 0.05, 0.95, 0.25, 0.05)
        stride = st.slider(
            "Analizar 1 de cada N fotogramas (video)",
            1,
            10,
            2,
            help="Un valor mayor acelera el analisis en CPU.",
        )

    if not Path(model_path).exists():
        st.error(f"No se encuentra el modelo en: {model_path}")
        st.stop()
    model = load_model(model_path)

    uploads = st.file_uploader(
        "Sube fotos y videos",
        type=IMAGE_TYPES + VIDEO_TYPES,
        accept_multiple_files=True,
    )

    if not uploads:
        st.markdown(
            '<div class="av-empty">ESPERANDO ARCHIVOS PARA ANALIZAR</div>',
            unsafe_allow_html=True,
        )
        return

    if not st.button("ANALIZAR"):
        return

    for uploaded in uploads:
        st.markdown(
            f'<div class="av-file">ARCHIVO // {escape(uploaded.name)}</div>',
            unsafe_allow_html=True,
        )
        extension = Path(uploaded.name).suffix.lower().lstrip(".")
        if extension in VIDEO_TYPES:
            process_video(model, uploaded, conf, stride)
        else:
            process_image(model, uploaded, conf)


if __name__ == "__main__":
    main()
