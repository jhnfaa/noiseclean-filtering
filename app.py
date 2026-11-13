"""NoiseClean: aplikasi demonstrasi filtering / noise reduction untuk tugas PSD.

Versi 2 - Wiener Filter domain frekuensi (STFT) agar SNR setelah filter >= 35 dB
pada noise 10 % RMS (target dosen). Wiener lokal (scipy.signal.wiener) tetap
tersedia sebagai pembanding / baseline.
"""

from io import BytesIO
import wave

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile
from scipy.signal import istft, stft, savgol_filter
import pywt

TARGET_SNR_DB = 35.0      # target dari dosen: SNR setelah filter minimal 35 dB
TARGET_NOISE_PCT = 10     # ... pada level noise minimal 10 % RMS

st.set_page_config(page_title="NoiseClean | PSD", page_icon="N", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #172033; --muted: #667085; --accent: #5b5ce2;
          --accent2: #818cf8; --line: #e5e9f2; --surface: #ffffff;
          --canvas: #f4f6fb;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stAppViewContainer"] { background: var(--canvas); }
        [data-testid="stHeader"] { background: rgba(244,246,251,.88); }
        [data-testid="stMainBlockContainer"] {
          max-width: 1480px; padding: 1.6rem 2.2rem 3rem;
        }
        /* Sidebar: compact dark control panel */
        [data-testid="stSidebar"] {
          background: #151d32; border-right: 1px solid #27324b;
        }
        [data-testid="stSidebar"] > div:first-child {
          background: transparent; padding-top: .8rem;
        }
        [data-testid="stSidebarUserContent"] { padding: 1rem 1rem 2rem; }
        [data-testid="stSidebar"] * { color: #e8ecf7 !important; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #aab5cc !important; }
        [data-testid="stSidebar"] hr { border-color: #34405b; }
        .control-title {
          margin: 1.35rem 0 .35rem; padding-top: .35rem;
          color: #9ca9c5 !important; font-size: .67rem;
          font-weight: 800; letter-spacing: .12em; text-transform: uppercase;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
          background: #202b44 !important; border: 1px solid #394762 !important;
          border-radius: 10px !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { color: #e8ecf7 !important; }
        [data-testid="stSidebar"] [role="slider"] { background: #9b9cff !important; }
        /* Header and hero */
        .topbar {
          display:flex; align-items:center; justify-content:space-between;
          padding: .85rem 1.1rem; margin: 0 0 1.5rem;
          background: #fff; border: 1px solid var(--line); border-radius: 14px;
          box-shadow: 0 3px 14px rgba(20,31,56,.04);
        }
        .brand { display:flex; align-items:center; gap:.7rem; color:var(--ink); font-weight:800; }
        .brand-mark {
          width:34px; height:34px; display:grid; place-items:center; border-radius:10px;
          color:#fff; font-size:.78rem; background:linear-gradient(135deg,#5558d9,#8b7cf6);
        }
        .topbar-right { color:#667085; font-size:.76rem; }
        .hero { padding: .3rem 0 1.15rem; margin:0; }
        .hero h1 { color:#172033; font-size:2.05rem; line-height:1.15; letter-spacing:-.055em; margin:.25rem 0 .55rem; }
        .hero p { color:#667085; font-size:.96rem; margin:0; max-width:850px; }
        .eyebrow { color:#5b5ce2; text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:850; }
        .hero-tag {
          display:inline-block; margin-top:.85rem; padding:.32rem .65rem; border-radius:7px;
          background:#eeefff; color:#4f46c8; font-size:.72rem; font-weight:750;
          border:1px solid #dedfff;
        }
        .experiment-strip {
          padding:.85rem 1rem; margin:.8rem 0 1.2rem; border-radius:12px;
          color:#475467; background:#eef2ff; border:1px solid #dfe5ff; font-size:.83rem;
        }
        .experiment-strip strong { color:#4f46c8; }
        /* Section hierarchy */
        h2, h3 { color:#202b43 !important; letter-spacing:-.025em; }
        h2 { font-size:1.35rem !important; }
        h3 { font-size:1.08rem !important; }
        p, label, [data-testid="stMarkdownContainer"] { color:inherit; }
        .section-title { color:#202b43; font-size:1.12rem; font-weight:800; margin:1.1rem 0 .2rem; }
        .section-subtitle { color:#667085; font-size:.84rem; margin-bottom:.7rem; }
        /* Metric cards */
        [data-testid="stMetric"] {
          background:#fff; border:1px solid var(--line); border-radius:14px;
          padding:1rem 1.05rem; box-shadow:0 4px 16px rgba(20,31,56,.045);
          min-height:125px;
        }
        [data-testid="stMetricLabel"] { color:#667085 !important; font-size:.78rem !important; font-weight:700; }
        [data-testid="stMetricValue"] { color:#202b43 !important; font-weight:800; letter-spacing:-.04em; }
        [data-testid="stMetricDelta"] { font-size:.75rem !important; }
        /* Tables and charts */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
          background:#fff; border:1px solid var(--line); border-radius:12px; overflow:hidden;
        }
        [data-testid="stPlotlyChart"] {
          background:#fff; border:1px solid var(--line); border-radius:14px;
          padding:.45rem .55rem .1rem; margin:.35rem 0 1rem;
          box-shadow:0 3px 12px rgba(20,31,56,.035);
        }
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
          gap:.35rem; border-bottom:1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
          height:44px; padding:0 1rem; color:#667085; font-weight:700;
          border-radius:9px 9px 0 0;
        }
        .stTabs [aria-selected="true"] { color:#4f46c8 !important; background:#eceeff; }
        /* Inputs and buttons */
        [data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
          border-radius:9px !important;
        }
        .stButton > button, .stDownloadButton > button {
          border:0 !important; border-radius:9px !important; padding:.55rem 1rem !important;
          background:linear-gradient(135deg,#5558d9,#7778ee) !important;
          color:#fff !important; font-weight:750 !important;
          box-shadow:0 4px 12px rgba(91,92,226,.18);
        }
        .stButton > button:hover, .stDownloadButton > button:hover { filter:brightness(1.06); }
        [data-testid="stExpander"] {
          background:#fff; border:1px solid var(--line); border-radius:12px;
        }
        [data-testid="stFileUploaderDropzone"] {
          background:#fff; border:1px dashed #b8c1d6; border-radius:12px;
        }
        .footer { text-align:center; color:#98a2b3; font-size:.75rem; padding:2rem 0 .5rem; }
        @media (max-width: 850px) {
          [data-testid="stMainBlockContainer"] { padding:1rem 1rem 2rem; }
          .hero h1 { font-size:1.55rem; }
          .topbar-right { display:none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Pemrosesan sinyal
# --------------------------------------------------------------------------- #
def to_float_audio(samples: np.ndarray) -> np.ndarray:
    """Normalise common WAV integer/float types to a mono float signal in [-1, 1]."""
    raw = np.asarray(samples)
    # Normalisasi HARUS dilakukan sebelum audio stereo dirata-ratakan.
    # Jika dirata-ratakan lebih dulu, dtype integer berubah menjadi float dengan
    # nilai PCM mentah (mis. ±32768) dan SNR akan salah terbaca sekitar 0 dB.
    if raw.dtype == np.uint8:  # WAV 8-bit bersifat unsigned, titik nol = 128
        values = (raw.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(raw.dtype, np.integer):
        scale = max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
        values = raw.astype(np.float64) / scale
    else:
        values = raw.astype(np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(values, -1.0, 1.0)


def make_test_signal(sample_rate: int, duration: float, frequency: float) -> np.ndarray:
    time = np.arange(int(sample_rate * duration)) / sample_rate
    # Komponen harmonik membuat bentuk gelombang dan spektrum lebih menarik untuk diamati.
    signal = (
        0.72 * np.sin(2 * np.pi * frequency * time)
        + 0.20 * np.sin(2 * np.pi * 2 * frequency * time)
        + 0.08 * np.sin(2 * np.pi * 3 * frequency * time)
    )
    return signal / np.max(np.abs(signal)) * 0.8


SAMPLE_KINDS = {
    "Flute (nada tunggal + vibrato)": "flute",
    "Organ (akor C-E-G)": "organ",
    "Dial tone (dua nada 697 + 1209 Hz)": "dial",
}


def make_sample_audio(kind: str, sample_rate: int, duration: float = 5.0) -> np.ndarray:
    """Contoh audio tonal (sustain) yang cocok didemokan dengan Wiener Filter."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    if kind == "flute":
        vibrato = 1 + 0.004 * np.sin(2 * np.pi * 5 * t)
        phase = 2 * np.pi * np.cumsum(880 * vibrato) / sample_rate
        x = np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)
    elif kind == "organ":
        x = sum(
            sum((1 / h) * np.sin(2 * np.pi * f * h * t) for h in range(1, 5))
            for f in (261.63, 329.63, 392.0)
        )
        x = x * np.minimum(t / 0.05, 1) * np.minimum((duration - t) / 0.1, 1)
    else:  # dial tone
        x = np.sin(2 * np.pi * 697 * t) + np.sin(2 * np.pi * 1209 * t)
    return x / np.max(np.abs(x)) * 0.8


def add_awgn(clean: np.ndarray, noise_percent: float, seed: int) -> np.ndarray:
    if noise_percent == 0:
        return clean.copy()
    signal_rms = np.sqrt(np.mean(clean**2))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, signal_rms * noise_percent / 100, clean.size)
    return np.clip(clean + noise, -1.0, 1.0)



def apply_wavelet_denoising(
    signal: np.ndarray, wavelet: str, level: int, threshold_scale: float
) -> np.ndarray:
    """DWT denoising dengan estimasi noise MAD dan soft thresholding."""
    x = np.asarray(signal, dtype=float)
    wave = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(x.size, wave.dec_len)
    used_level = max(1, min(int(level), max_level)) if max_level >= 1 else 1
    coeffs = pywt.wavedec(x, wavelet, mode="symmetric", level=used_level)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
    threshold = float(threshold_scale) * sigma * np.sqrt(2 * np.log(max(x.size, 2)))
    denoised = [coeffs[0]]
    denoised.extend(pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:])
    result = pywt.waverec(denoised, wavelet, mode="symmetric")
    return result[:x.size]


def apply_savgol(signal: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing dengan penyesuaian window agar valid."""
    x = np.asarray(signal, dtype=float)
    win = min(int(window), x.size if x.size % 2 else x.size - 1)
    if win < 3:
        return x.copy()
    if win % 2 == 0:
        win -= 1
    order = min(int(polyorder), win - 1)
    return savgol_filter(x, window_length=win, polyorder=order, mode="interp")


def estimate_noise_psd(power: np.ndarray) -> float:
    """Estimasi PSD noise putih dari spektrogram daya.

    Median per-bin sepanjang waktu tahan terhadap nada (nada hanya menempati sedikit
    bin/frame). Median distribusi eksponensial = rata-rata * ln2, sehingga dibagi ln2.
    Persentil rendah antar-frekuensi menghindari bias dari bin yang berisi sinyal.
    """
    per_bin = np.median(power, axis=1) / np.log(2)
    return float(np.percentile(per_bin, 40))


def apply_wiener_stft(signal: np.ndarray, sample_rate: int, nperseg: int, alpha: float) -> np.ndarray:
    """Wiener Filter domain frekuensi (STFT) dengan a-priori SNR 'decision-directed'.

    H(f,t) = xi / (1 + xi),  xi = a-priori SNR = P_sinyal / P_noise
    Y_hat(f,t) = H(f,t) * Y(f,t)  ->  ISTFT  ->  sinyal terestimasi
    """
    frame = int(min(nperseg, 2 ** int(np.floor(np.log2(max(signal.size, 64))))))
    overlap = frame * 3 // 4
    _, _, spec = stft(signal, fs=sample_rate, window="hann", nperseg=frame, noverlap=overlap)
    power = np.abs(spec) ** 2
    noise_psd = max(estimate_noise_psd(power), 1e-20)

    result = np.empty_like(spec)
    previous = np.maximum(power[:, 0] - noise_psd, 0.0)
    for k in range(spec.shape[1]):
        prior_snr = alpha * previous / noise_psd + (1 - alpha) * np.maximum(power[:, k] / noise_psd - 1, 0.0)
        gain = prior_snr / (1 + prior_snr)
        result[:, k] = gain * spec[:, k]
        previous = np.abs(result[:, k]) ** 2

    _, filtered = istft(result, fs=sample_rate, window="hann", nperseg=frame, noverlap=overlap)
    filtered = filtered[: signal.size]
    if filtered.size < signal.size:
        filtered = np.pad(filtered, (0, signal.size - filtered.size))
    return filtered


def snr_db(reference: np.ndarray, tested: np.ndarray) -> float:
    error = reference - tested
    numerator = np.mean(reference**2)
    denominator = np.mean(error**2)
    if denominator < 1e-15:
        return float("inf")
    return 10 * np.log10(numerator / denominator)


def wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(signal, -1, 1)
    pcm = (clipped * 32767).astype(np.int16)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
#  Grafik
# --------------------------------------------------------------------------- #
def plot_waveforms(clean: np.ndarray, noisy: np.ndarray, filtered: np.ndarray, sample_rate: int) -> go.Figure:
    max_points = min(clean.size, int(sample_rate * 0.025))
    time_ms = np.arange(max_points) / sample_rate * 1000
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=time_ms, y=clean[:max_points], mode="lines", name="Sinyal awal", line={"color": "#2563eb", "width": 3}, hovertemplate="Waktu: %{x:.2f} ms<br>Amplitudo: %{y:.3f}<extra>Sinyal awal</extra>"))
    figure.add_trace(go.Scatter(x=time_ms, y=noisy[:max_points], mode="lines", name="Ber-noise", line={"color": "#fb7185", "width": 1.4}, opacity=.72, hovertemplate="Waktu: %{x:.2f} ms<br>Amplitudo: %{y:.3f}<extra>Ber-noise</extra>"))
    figure.add_trace(go.Scatter(x=time_ms, y=filtered[:max_points], mode="lines", name="Hasil filter", line={"color": "#0f9f8a", "width": 3}, hovertemplate="Waktu: %{x:.2f} ms<br>Amplitudo: %{y:.3f}<extra>Hasil filter</extra>"))
    figure.update_layout(
        title={"text": "Domain waktu - 25 ms pertama", "x": .02, "xanchor": "left"},
        height=360, margin={"l": 54, "r": 20, "t": 62, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.35)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": .98},
        hovermode="x unified",
    )
    figure.update_xaxes(title="Waktu (ms)", gridcolor="rgba(148,163,184,.35)", zeroline=False)
    figure.update_yaxes(title="Amplitudo", gridcolor="rgba(148,163,184,.35)", zerolinecolor="#cbd5e1")
    return figure


def plot_spectrum(clean: np.ndarray, noisy: np.ndarray, filtered: np.ndarray, sample_rate: int) -> go.Figure:
    n = min(clean.size, 32768)
    window = np.hanning(n)
    frequencies = np.fft.rfftfreq(n, 1 / sample_rate)

    def magnitude_db(values: np.ndarray) -> np.ndarray:
        return 20 * np.log10(np.maximum(np.abs(np.fft.rfft(values[:n] * window)), 1e-10))

    figure = go.Figure()
    for values, name, color, width, opacity in [(clean, "Awal", "#2563eb", 2.4, 1), (noisy, "Ber-noise", "#fb7185", 1, .72), (filtered, "Terfilter", "#0f9f8a", 2.4, 1)]:
        figure.add_trace(go.Scatter(x=frequencies, y=magnitude_db(values), mode="lines", name=name, line={"color": color, "width": width}, opacity=opacity))
    figure.update_layout(
        title={"text": "Spektrum frekuensi", "x": .02, "xanchor": "left"},
        height=400, margin={"l": 54, "r": 20, "t": 62, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.35)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": .98},
        hovermode="x unified",
    )
    figure.update_xaxes(title="Frekuensi (Hz)", range=[0, min(8000, sample_rate / 2)], gridcolor="rgba(148,163,184,.35)", zeroline=False)
    figure.update_yaxes(title="Magnitudo (dB)", range=[-20, 75], gridcolor="rgba(148,163,184,.35)", zeroline=False)
    return figure


def plot_quality_comparison(before: float, after: float) -> go.Figure:
    """Perbandingan SNR sebelum/sesudah filter, lengkap dengan garis target dosen."""
    values = [before, after]
    improvement = after - before
    figure = go.Figure(go.Bar(x=["Sebelum filter", "Sesudah filter"], y=values, marker_color=["#fb7185", "#14b8a6"], text=[f"{before:.2f} dB", f"{after:.2f} dB"], textposition="outside", textfont={"size": 14, "color": "#0f172a"}, hovertemplate="%{x}<br>SNR: %{y:.2f} dB<extra></extra>"))
    figure.add_hline(y=TARGET_SNR_DB, line_dash="dash", line_color="#7c3aed", annotation_text=f"Target {TARGET_SNR_DB:.0f} dB", annotation_position="top left", annotation_font_color="#7c3aed")
    figure.update_layout(
        title={"text": f"Kualitas sinyal · perubahan {improvement:+.2f} dB", "x": .02, "xanchor": "left"},
        height=360, margin={"l": 26, "r": 20, "t": 62, "b": 46},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.35)", showlegend=False,
    )
    figure.update_yaxes(title="SNR (dB)", range=[min(-5, min(values) - 3), max(max(values), TARGET_SNR_DB) + 6], gridcolor="rgba(148,163,184,.35)", zeroline=False)
    figure.update_xaxes(fixedrange=True)
    return figure




def plot_all_waveforms(clean, noisy, outputs, sample_rate):
    count = min(clean.size, int(sample_rate * 0.025))
    time_ms = np.arange(count) / sample_rate * 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_ms, y=clean[:count], name="Clean reference",
                             line={"color": "#2563eb", "width": 2.5}))
    fig.add_trace(go.Scatter(x=time_ms, y=noisy[:count], name="Noisy input",
                             line={"color": "#fb7185", "width": 1.2}, opacity=0.7))
    colors = ["#0f9f8a", "#8b5cf6", "#f59e0b"]
    for (name, values), color in zip(outputs.items(), colors):
        fig.add_trace(go.Scatter(x=time_ms, y=values[:count], name=name,
                                 line={"width": 2, "color": color}))
    fig.update_layout(title="Perbandingan waveform (25 ms pertama)", height=420,
                      xaxis_title="Waktu (ms)", yaxis_title="Amplitudo",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.35)",
                      legend={"orientation": "h", "y": 1.12})
    return fig


def plot_all_spectra(clean, noisy, outputs, sample_rate):
    n = min(clean.size, 32768)
    window = np.hanning(n)
    freq = np.fft.rfftfreq(n, 1 / sample_rate)

    def db(x):
        return 20 * np.log10(np.maximum(np.abs(np.fft.rfft(x[:n] * window)), 1e-10))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=db(clean), name="Clean reference",
                             line={"color": "#2563eb", "width": 2.5}))
    fig.add_trace(go.Scatter(x=freq, y=db(noisy), name="Noisy input",
                             line={"color": "#fb7185", "width": 1.2}, opacity=0.65))
    colors = ["#0f9f8a", "#8b5cf6", "#f59e0b"]
    for (name, values), color in zip(outputs.items(), colors):
        fig.add_trace(go.Scatter(x=freq, y=db(values), name=name,
                                 line={"width": 2, "color": color}))
    fig.update_layout(title="Perbandingan spektrum FFT", height=420,
                      xaxis_title="Frekuensi (Hz)", yaxis_title="Magnitudo (dB)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.35)",
                      legend={"orientation": "h", "y": 1.12})
    fig.update_xaxes(range=[0, min(8000, sample_rate / 2)])
    return fig


def calculate_metrics(reference, tested, noisy):
    error = reference - tested
    mse = float(np.mean(error ** 2))
    return {
        "SNR (dB)": float(snr_db(reference, tested)),
        "RMSE": float(np.sqrt(mse)),
        "Peningkatan SNR (dB)": float(snr_db(reference, tested) - snr_db(reference, noisy)),
    }

# --------------------------------------------------------------------------- #
#  NoiseClean - eksperimen tiga metode sesuai ruang lingkup artikel
# --------------------------------------------------------------------------- #
import pandas as pd

inject_css()
st.markdown(
    """<div class="topbar"><div class="brand"><div class="brand-mark">NC</div>NoiseClean</div>
    <div class="topbar-right">PSD Lab · Kelompok 6</div></div>
    <div class="hero"><div class="eyebrow">Digital Signal Processing Lab</div>
    <h1>Analisis Perbandingan Reduksi Derau Sinyal Audio</h1>
    <p>Perbandingan Wiener Filter, Savitzky–Golay, dan Wavelet Denoising berdasarkan SNR dan RMSE.</p>
    <span class="hero-tag">Three-Method Comparative Experiment</span></div>""",
    unsafe_allow_html=True,
)

METHOD_WIENER = "Wiener Filter (STFT)"
METHOD_SG = "Savitzky–Golay Filter"
METHOD_WAVELET = "Wavelet Denoising"
NOISE_SCENARIOS = [1, 10]

with st.sidebar:
    st.header("Control Lab")
    st.caption("Atur sumber sinyal dan parameter eksperimen.")
    st.markdown('<div class="control-title">Sinyal input</div>', unsafe_allow_html=True)
    source = st.radio("Sumber sinyal", ["Sinyal sintetis", "Contoh audio referensi", "Unggah WAV"])
    uploaded = None
    sample_kind = "flute"
    if source == "Sinyal sintetis":
        sample_rate = st.selectbox("Sampling rate", [8000, 16000, 44100], index=1)
        duration = st.slider("Durasi (detik)", 1.0, 10.0, 3.0, 0.5)
        frequency = st.slider("Frekuensi fundamental (Hz)", 100, 2000, 440, 10)
    elif source == "Contoh audio referensi":
        sample_kind = SAMPLE_KINDS[st.selectbox("Jenis contoh audio", list(SAMPLE_KINDS))]
        sample_rate = st.selectbox("Sampling rate", [16000, 44100], index=1)
        duration, frequency = 5.0, 440
    else:
        uploaded = st.file_uploader("File audio WAV", type=["wav"])
        sample_rate, duration, frequency = 16000, 3.0, 440

    st.markdown('<div class="control-title">Parameter eksperimen</div>', unsafe_allow_html=True)
    random_seed = st.number_input("Seed noise (konsisten)", min_value=0, value=6, step=1)
    st.caption("Aplikasi otomatis menguji dua skenario: AWGN 1% dan 10%. Ketiga metode menerima noisy input yang sama pada setiap skenario.")

    st.markdown('<div class="control-title">Parameter Wiener STFT</div>', unsafe_allow_html=True)
    frame_size = st.select_slider("Ukuran frame STFT (sampel)", options=[512, 1024, 2048, 4096], value=4096)
    alpha = st.slider("Kehalusan a-priori SNR (α)", 0.90, 0.995, 0.98, 0.005)

    st.markdown('<div class="control-title">Parameter Wavelet</div>', unsafe_allow_html=True)
    wavelet_name = st.selectbox("Wavelet family", ["db4", "sym4", "coif1"], index=0)
    wavelet_level = st.slider("Decomposition level", 1, 6, 3)
    threshold_scale = st.slider("Threshold scale", 0.25, 2.00, 1.00, 0.05)

    st.markdown('<div class="control-title">Parameter Savitzky–Golay</div>', unsafe_allow_html=True)
    sg_window = st.slider("Window length (ganjil)", 5, 101, 11, 2)
    sg_order = st.slider("Polynomial order", 2, 5, 3)

try:
    if source == "Unggah WAV":
        if uploaded is None:
            st.info("Unggah file WAV dari panel kiri, atau pilih sinyal sintetis untuk memulai.")
            st.stop()
        sample_rate, raw = wavfile.read(uploaded)
        clean_signal = to_float_audio(raw)
        maximum_samples = int(sample_rate * 30)
        if clean_signal.size > maximum_samples:
            clean_signal = clean_signal[:maximum_samples]
            st.warning("Audio dibatasi hingga 30 detik agar visualisasi tetap responsif.")
    elif source == "Contoh audio referensi":
        clean_signal = make_sample_audio(sample_kind, sample_rate, duration)
    else:
        clean_signal = make_test_signal(sample_rate, duration, frequency)

    if clean_signal.size < 100:
        st.error("Sinyal terlalu pendek untuk diproses.")
        st.stop()

    # Run both fixed noise scenarios. Each method receives the exact same noisy signal.
    scenario_results = {}
    all_metric_rows = []
    for noise_percent in NOISE_SCENARIOS:
        noisy = add_awgn(clean_signal, noise_percent, int(random_seed))
        outputs = {
            METHOD_WIENER: apply_wiener_stft(noisy, int(sample_rate), frame_size, alpha),
            METHOD_SG: apply_savgol(noisy, sg_window, sg_order),
            METHOD_WAVELET: apply_wavelet_denoising(noisy, wavelet_name, wavelet_level, threshold_scale),
        }
        scenario_results[noise_percent] = {"noisy": noisy, "outputs": outputs}
        baseline_snr = snr_db(clean_signal, noisy)
        for method_name, result in outputs.items():
            row = calculate_metrics(clean_signal, result, noisy)
            all_metric_rows.append({
                "Skenario noise": f"{noise_percent}%",
                "Metode": method_name,
                "SNR (dB)": row["SNR (dB)"],
                "RMSE": row["RMSE"],
                "Peningkatan SNR (dB)": row["Peningkatan SNR (dB)"],
            })

    source_name = {
        "Sinyal sintetis": "Sinyal sintetis",
        "Contoh audio referensi": "Contoh audio referensi",
        "Unggah WAV": "Audio WAV",
    }[source]

    st.markdown(
        f"""<div class="experiment-strip"><strong>Eksperimen aktif</strong> &nbsp; {source_name}
        · {int(sample_rate):,} Hz · {clean_signal.size / sample_rate:.1f} detik
        &nbsp; | &nbsp; Seed {int(random_seed)} &nbsp; | &nbsp; 3 metode × 2 skenario noise</div>""",
        unsafe_allow_html=True,
    )

    scenario_tab1, scenario_tab2 = st.tabs(["Skenario noise 1%", "Skenario noise 10%"])
    for tab, noise_percent in zip([scenario_tab1, scenario_tab2], NOISE_SCENARIOS):
        with tab:
            noisy = scenario_results[noise_percent]["noisy"]
            outputs = scenario_results[noise_percent]["outputs"]
            before = snr_db(clean_signal, noisy)
            rows = [r for r in all_metric_rows if r["Skenario noise"] == f"{noise_percent}%"]
            metrics_df = pd.DataFrame(rows)

            st.markdown(f"### Hasil pengujian noise {noise_percent}% RMS")
            st.caption("Ketiga metode diproses dari satu noisy input yang sama dengan seed identik.")
            cols = st.columns(3)
            for col, row in zip(cols, rows):
                with col:
                    st.metric(row["Metode"], f'{row["SNR (dB)"]:.2f} dB',
                              f'RMSE {row["RMSE"]:.6f}')
            st.markdown("#### Tabel perbandingan metode")
            st.dataframe(
                metrics_df[["Metode", "SNR (dB)", "RMSE", "Peningkatan SNR (dB)"]].style.format({
                    "SNR (dB)": "{:.2f}", "RMSE": "{:.6f}",
                    "Peningkatan SNR (dB)": "{:+.2f}"
                }),
                use_container_width=True, hide_index=True
            )

            snr_fig = go.Figure()
            snr_fig.add_trace(go.Bar(
                x=[r["Metode"] for r in rows],
                y=[r["SNR (dB)"] for r in rows],
                text=[f'{r["SNR (dB)"]:.2f} dB' for r in rows],
                textposition="outside",
                marker_color=["#0f9f8a", "#8b5cf6", "#f59e0b"],
            ))
            snr_fig.update_layout(title=f"Perbandingan SNR — noise {noise_percent}%",
                                  yaxis_title="SNR (dB)", height=340,
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(255,255,255,.35)")
            st.plotly_chart(snr_fig, use_container_width=True, config={"displaylogo": False})

            rmse_fig = go.Figure()
            rmse_fig.add_trace(go.Bar(
                x=[r["Metode"] for r in rows],
                y=[r["RMSE"] for r in rows],
                text=[f'{r["RMSE"]:.6f}' for r in rows],
                textposition="outside",
                marker_color=["#0f9f8a", "#8b5cf6", "#f59e0b"],
            ))
            rmse_fig.update_layout(title=f"Perbandingan RMSE — noise {noise_percent}%",
                                   yaxis_title="RMSE", height=340,
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(255,255,255,.35)")
            st.plotly_chart(rmse_fig, use_container_width=True, config={"displaylogo": False})

            st.markdown("#### Perbandingan waveform")
            st.plotly_chart(plot_all_waveforms(clean_signal, noisy, outputs, int(sample_rate)),
                            use_container_width=True, config={"displaylogo": False})
            st.markdown("#### Perbandingan spektrum FFT")
            st.plotly_chart(plot_all_spectra(clean_signal, noisy, outputs, int(sample_rate)),
                            use_container_width=True, config={"displaylogo": False})

            with st.expander("Dengarkan dan unduh hasil audio"):
                audio_cols = st.columns(3)
                for col, (name, result) in zip(audio_cols, outputs.items()):
                    with col:
                        st.markdown(f"**{name}**")
                        st.audio(wav_bytes(result, sample_rate), format="audio/wav")
                        safe_name = name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                        st.download_button(
                            f"Unduh {name}",
                            wav_bytes(result, sample_rate),
                            file_name=f"noiseclean_{safe_name}_noise_{noise_percent}pct.wav",
                            mime="audio/wav",
                            key=f"download_{noise_percent}_{safe_name}",
                        )

    st.markdown("### Rekap seluruh skenario")
    full_df = pd.DataFrame(all_metric_rows)
    st.dataframe(full_df.style.format({
        "SNR (dB)": "{:.2f}", "RMSE": "{:.6f}",
        "Peningkatan SNR (dB)": "{:+.2f}"
    }), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Unduh seluruh hasil eksperimen (CSV)",
        full_df.to_csv(index=False).encode("utf-8"),
        file_name="noiseclean_results_1pct_10pct.csv",
        mime="text/csv",
    )

    with st.expander("Metode dan interpretasi"):
        st.markdown("#### Wiener Filter berbasis STFT")
        st.latex(r"H(f,t)=\frac{\xi(f,t)}{1+\xi(f,t)}")
        st.write("Wiener Filter STFT mengestimasi gain berdasarkan rasio SNR a-priori pada setiap bin frekuensi dan frame.")
        st.markdown("#### Wavelet Denoising")
        st.write("Sinyal ditransformasikan menggunakan DWT. Koefisien detail diberi soft threshold berdasarkan estimasi noise MAD, kemudian direkonstruksi dengan inverse DWT.")
        st.markdown("#### Savitzky–Golay Filter")
        st.write("Filter melakukan pemulusan melalui fitting polinomial lokal pada jendela sampel.")
        st.markdown("#### Metrik evaluasi")
        st.write("SNR dihitung terhadap clean reference; nilai lebih tinggi menunjukkan error relatif lebih kecil. RMSE adalah akar rata-rata kuadrat error; nilai lebih rendah menunjukkan hasil lebih dekat ke clean reference.")
        st.info("Untuk perbandingan apple-to-apple, jangan mengubah sumber sinyal, sampling rate, durasi, atau seed ketika membandingkan ketiga metode pada satu skenario.")
except Exception as error:
    st.error(f"Tidak dapat memproses sinyal: {error}")
    st.exception(error)

st.markdown('<div class="footer">NoiseClean · Aplikasi Pengolahan Sinyal Digital · Kelompok 6</div>', unsafe_allow_html=True)
