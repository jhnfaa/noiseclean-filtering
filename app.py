
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pywt
import streamlit as st

from scipy.io import wavfile
from scipy.signal import stft, istft, savgol_filter


# =========================================================
# 1. KONFIGURASI APLIKASI
# Harus menjadi perintah Streamlit pertama.
# =========================================================

st.set_page_config(
    page_title="NoiseClean | PSD",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PARAMETER EKSPERIMEN TETAP
# =========================================================

WIENER_FRAME = 512
WIENER_ALPHA = 0.98

SAVGOL_WINDOW = 31
SAVGOL_POLYORDER = 3

WAVELET_NAME = "db4"
WAVELET_LEVEL = 3
WAVELET_THRESHOLD = 1.00

METHOD_ORDER = [
    "Wiener Filter (STFT)",
    "Savitzky-Golay Filter",
    "Wavelet Denoising"
]

CHART_COLORS = [
    "#3157D5",
    "#0D9488",
    "#D97706"
]


# =========================================================
# 3. CSS / TAMPILAN
# =========================================================

def inject_css():
    st.markdown("""
    <style>

    .stApp {
        background-color: #F3F5FB;
        color: #172033;
    }

    [data-testid="stSidebar"] {
        background-color: #151D32;
    }

    [data-testid="stSidebar"] * {
        color: #F9FAFB;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #B7C2D8;
    }

    [data-testid="stSidebar"] label {
        color: #F9FAFB !important;
    }

    [data-testid="stSidebar"] input {
        color: #172033 !important;
        background-color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] {
        color: #172033 !important;
        background-color: #FFFFFF !important;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E1E7F0;
        border-radius: 14px;
        padding: 18px;
        min-height: 120px;
    }

    [data-testid="stPlotlyChart"] {
        background-color: #FFFFFF;
        border: 1px solid #E1E7F0;
        border-radius: 14px;
        padding: 8px;
    }

    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border-radius: 12px;
    }

    .topbar {
        background-color: #FFFFFF;
        padding: 17px 22px;
        border: 1px solid #E1E7F0;
        border-radius: 14px;
        margin-bottom: 28px;
        font-weight: 800;
        font-size: 18px;
        color: #172033;
    }

    .hero-label {
        color: #5B5CE2;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 850;
        color: #172033;
        margin-bottom: 10px;
    }

    .hero-description {
        color: #667085;
        font-size: 15px;
        margin-bottom: 22px;
    }

    .section-label {
        color: #AAB8D1;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .fixed-method {
        background-color: #202D49;
        border: 1px solid #394967;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
        font-size: 13px;
    }

    .fixed-method strong {
        display: block;
        color: #FFFFFF;
        margin-bottom: 7px;
    }

    .fixed-method span {
        display: block;
        color: #D6DEEC;
        margin-top: 4px;
    }

    .footer {
        text-align: center;
        color: #98A2B3;
        padding: 30px 0;
        font-size: 12px;
    }

    </style>
    """, unsafe_allow_html=True)


# =========================================================
# 4. FUNGSI PENGOLAHAN SINYAL
# =========================================================

def to_float_audio(samples):
    raw = np.asarray(samples)

    # Mengubah stereo menjadi mono
    if raw.ndim == 2:
        raw = raw.astype(float).mean(axis=1)

    # Normalisasi berdasarkan tipe data WAV
    if raw.dtype == np.uint8:
        values = (raw.astype(float) - 128) / 128

    elif np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        scale = max(abs(info.min), info.max)
        values = raw.astype(float) / scale

    else:
        values = raw.astype(float)

    return np.nan_to_num(values)


def normalize_signal(x):
    x = np.asarray(x, dtype=float)

    peak = np.max(np.abs(x))

    if peak == 0:
        raise ValueError(
            "Sinyal kosong atau memiliki amplitudo nol."
        )

    return x / peak * 0.8


def make_test_signal(fs, duration, frequency):
    t = np.arange(int(fs * duration)) / fs

    x = (
        0.72 * np.sin(2 * np.pi * frequency * t)
        + 0.20 * np.sin(2 * np.pi * 2 * frequency * t)
        + 0.08 * np.sin(2 * np.pi * 3 * frequency * t)
    )

    return normalize_signal(x)


def make_sample_audio(kind, fs, duration):
    t = np.arange(int(fs * duration)) / fs

    if kind == "Flute":
        vibrato = 1 + 0.004 * np.sin(2 * np.pi * 5 * t)

        phase = 2 * np.pi * np.cumsum(
            880 * vibrato
        ) / fs

        x = (
            np.sin(phase)
            + 0.25 * np.sin(2 * phase)
            + 0.08 * np.sin(3 * phase)
        )

    elif kind == "Organ":
        x = sum(
            sum(
                (1 / h) * np.sin(
                    2 * np.pi * f * h * t
                )
                for h in range(1, 5)
            )
            for f in (261.63, 329.63, 392.0)
        )

        envelope = np.minimum(t / 0.05, 1)
        envelope *= np.minimum(
            (duration - t) / 0.1,
            1
        )

        x *= envelope

    else:
        x = (
            np.sin(2 * np.pi * 697 * t)
            + np.sin(2 * np.pi * 1209 * t)
        )

    return normalize_signal(x)


def add_awgn(clean, noise_percent, seed):
    """
    Menambahkan Gaussian noise berdasarkan persentase
    RMS sinyal bersih.
    """

    if noise_percent <= 0:
        return clean.copy()

    signal_rms = np.sqrt(np.mean(clean ** 2))

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        0,
        signal_rms * noise_percent / 100,
        clean.size
    )

    return clean + noise


# =========================================================
# 5. WIENER FILTER BERBASIS STFT
# =========================================================

def apply_wiener(signal, fs):
    x = np.asarray(signal, dtype=float)

    nperseg = min(WIENER_FRAME, len(x))

    if nperseg < 16:
        raise ValueError(
            "Sinyal terlalu pendek untuk Wiener STFT."
        )

    noverlap = nperseg // 2

    frequencies, times, Z = stft(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros"
    )

    power = np.abs(Z) ** 2

    # Estimasi daya noise
    noise_power = np.percentile(
        power,
        20,
        axis=1,
        keepdims=True
    )

    # Wiener gain
    gain = np.maximum(
        1 - WIENER_ALPHA * noise_power /
        (power + 1e-12),
        0.05
    )

    filtered_Z = Z * gain

    _, result = istft(
        filtered_Z,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True
    )

    result = result[:len(x)]

    if len(result) < len(x):
        result = np.pad(
            result,
            (0, len(x) - len(result))
        )

    return result


# =========================================================
# 6. SAVITZKY-GOLAY FILTER
# =========================================================

def apply_savgol(signal):
    x = np.asarray(signal, dtype=float)

    window = min(
        SAVGOL_WINDOW,
        len(x) if len(x) % 2 else len(x) - 1
    )

    if window < 3:
        return x.copy()

    if window % 2 == 0:
        window -= 1

    order = min(
        SAVGOL_POLYORDER,
        window - 1
    )

    return savgol_filter(
        x,
        window_length=window,
        polyorder=order,
        mode="interp"
    )


# =========================================================
# 7. WAVELET DENOISING
# =========================================================

def apply_wavelet(signal):
    x = np.asarray(signal, dtype=float)

    wave = pywt.Wavelet(WAVELET_NAME)

    max_level = pywt.dwt_max_level(
        len(x),
        wave.dec_len
    )

    if max_level < 1:
        raise ValueError(
            "Sinyal terlalu pendek untuk Wavelet Denoising."
        )

    level = min(WAVELET_LEVEL, max_level)

    coeffs = pywt.wavedec(
        x,
        WAVELET_NAME,
        mode="symmetric",
        level=level
    )

    detail = coeffs[-1]

    sigma = (
        np.median(
            np.abs(detail - np.median(detail))
        ) / 0.6745
    )

    threshold = (
        WAVELET_THRESHOLD
        * sigma
        * np.sqrt(2 * np.log(max(len(x), 2)))
    )

    filtered_coeffs = [coeffs[0]]

    filtered_coeffs.extend(
        pywt.threshold(
            c,
            threshold,
            mode="soft"
        )
        for c in coeffs[1:]
    )

    result = pywt.waverec(
        filtered_coeffs,
        WAVELET_NAME,
        mode="symmetric"
    )

    return result[:len(x)]


# =========================================================
# 8. METRIK EVALUASI
# =========================================================

def calculate_metrics(clean, noisy, filtered):
    clean = np.asarray(clean, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    filtered = np.asarray(filtered, dtype=float)

    n = min(
        len(clean),
        len(noisy),
        len(filtered)
    )

    clean = clean[:n]
    noisy = noisy[:n]
    filtered = filtered[:n]

    noise_before = clean - noisy
    noise_after = clean - filtered

    power_signal = np.mean(clean ** 2)

    power_before = np.mean(noise_before ** 2)
    power_after = np.mean(noise_after ** 2)

    snr_before = 10 * np.log10(
        power_signal / (power_before + 1e-15)
    )

    snr_after = 10 * np.log10(
        power_signal / (power_after + 1e-15)
    )

    rmse = np.sqrt(
        np.mean((clean - filtered) ** 2)
    )

    improvement = snr_after - snr_before

    return {
        "SNR input (dB)": snr_before,
        "SNR (dB)": snr_after,
        "RMSE": rmse,
        "Peningkatan SNR (dB)": improvement
    }


# =========================================================
# 9. GRAFIK SINYAL
# =========================================================



def make_signal_chart(
    time,
    clean,
    noisy,
    filtered,
    title
):
    # Batasi jumlah titik agar grafik tetap ringan
    max_points = 5000

    step = max(
        1,
        int(np.ceil(len(time) / max_points))
    )

    # Ambil cuplikan 25 milidetik
    view_seconds = 0.025
    mask = time <= view_seconds

    if np.sum(mask) < 2:
        mask = np.ones(len(time), dtype=bool)

    t = time[mask][::step]
    clean_plot = clean[mask][::step]
    noisy_plot = noisy[mask][::step]
    filtered_plot = filtered[mask][::step]

    # Buat satu grafik dengan tiga garis
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t,
        y=clean_plot,
        mode="lines",
        name="Sinyal bersih",
        line=dict(
            color="#0D9488",
            width=2
        ),
        opacity=0.95
    ))

    fig.add_trace(go.Scatter(
        x=t,
        y=noisy_plot,
        mode="lines",
        name="Sinyal noisy",
        line=dict(
            color="#E69A36",
            width=1.3
        ),
        opacity=0.75
    ))

    fig.add_trace(go.Scatter(
        x=t,
        y=filtered_plot,
        mode="lines",
        name="Sinyal hasil filter",
        line=dict(
            color="#3157D5",
            width=2
        ),
        opacity=0.95
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(
                family="Arial",
                size=19,
                color="#172033"
            )
        ),

        template="plotly_white",
        height=520,

        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        font=dict(
            family="Arial",
            size=12,
            color="#344054"
        ),

        hovermode="x unified",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)"
        ),

        margin=dict(
            l=65,
            r=35,
            t=110,
            b=70
        )
    )

    fig.update_xaxes(
        title_text="Waktu (detik)",
        showgrid=True,
        gridcolor="#E9EDF4",
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1"
    )

    fig.update_yaxes(
        title_text="Amplitudo",
        showgrid=True,
        gridcolor="#E9EDF4",
        zeroline=True,
        zerolinecolor="#CBD5E1"
    )

    return fig


# =========================================================
# 10. GRAFIK METRIK SATU FILTER
# =========================================================

def make_metric_chart(name, value, title, suffix=""):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[name],
        y=[value],
        text=[f"{value:.4f}{suffix}"],
        textposition="outside",
        marker_color="#3157D5",
        width=0.45
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.05,
            font=dict(size=15)
        ),
        template="plotly_white",
        height=320,
        showlegend=False,
        yaxis_title=title,
        font=dict(
            family="Arial",
            size=12,
            color="#344054"
        ),
        margin=dict(
            l=40,
            r=25,
            t=65,
            b=55
        )
    )

    return fig


# =========================================================
# 11. GRAFIK PERBANDINGAN TIGA METODE
# =========================================================

def make_comparison_chart(
    dataframe,
    metric,
    title,
    y_title,
    suffix="",
    decimals=2,
    zero_line=False
):
    fig = go.Figure()

    values = dataframe[metric].astype(float)

    text_values = [
        f"{value:+.{decimals}f}{suffix}"
        if metric == "Peningkatan SNR (dB)"
        else f"{value:.{decimals}f}{suffix}"
        for value in values
    ]

    fig.add_trace(go.Bar(
        x=dataframe["Metode"].astype(str),
        y=values,
        text=text_values,
        textposition="outside",
        marker_color=CHART_COLORS,
        marker_line=dict(
            width=0
        ),
        cliponaxis=False
    ))

    if zero_line:
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#64748B",
            line_width=1
        )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
            font=dict(size=17)
        ),
        xaxis_title="Metode denoising",
        yaxis_title=y_title,
        template="plotly_white",
        height=440,
        showlegend=False,
        font=dict(
            family="Arial",
            size=12,
            color="#344054"
        ),
        margin=dict(
            l=60,
            r=40,
            t=80,
            b=100
        ),
        xaxis=dict(
            tickangle=0,
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E8ECF3",
            zeroline=False
        )
    )

    return fig


# =========================================================
# 12. TAMPILAN SIDEBAR
# =========================================================

inject_css()

with st.sidebar:

    st.markdown("## 🎧 NoiseClean")
    st.caption("Digital Signal Processing Laboratory")

    st.markdown("---")

    st.markdown(
        '<div class="section-label">SUMBER SINYAL</div>',
        unsafe_allow_html=True
    )

    source = st.radio(
        "Pilih sumber sinyal",
        [
            "Sinyal sintetis",
            "Contoh audio referensi",
            "Unggah WAV"
        ],
        label_visibility="collapsed"
    )

    sample_rate = 16000
    duration = 5.0
    frequency = 440
    uploaded = None
    sample_kind = "Flute"

    if source == "Sinyal sintetis":

        sample_rate = st.selectbox(
            "Sampling rate (Hz)",
            [8000, 16000, 44100],
            index=1
        )

        duration = st.slider(
            "Durasi (detik)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        )

        frequency = st.slider(
            "Frekuensi fundamental (Hz)",
            min_value=100,
            max_value=min(2000, sample_rate // 2 - 1),
            value=min(440, sample_rate // 2 - 1),
            step=10
        )

    elif source == "Contoh audio referensi":

        sample_kind = st.selectbox(
            "Jenis contoh audio",
            ["Flute", "Organ", "Dial tone"]
        )

        sample_rate = st.selectbox(
            "Sampling rate (Hz)",
            [16000, 44100],
            index=0
        )

        duration = 5.0

    else:

        uploaded = st.file_uploader(
            "Unggah file audio WAV",
            type=["wav"]
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-label">TINGKAT NOISE</div>',
        unsafe_allow_html=True
    )

    noise_percent = st.slider(
        "Persentase noise (%)",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )

    st.caption(
        "Noise yang sama digunakan untuk ketiga metode."
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-label">PARAMETER EKSPERIMEN</div>',
        unsafe_allow_html=True
    )

    random_seed = st.number_input(
        "Seed noise (konsisten)",
        min_value=0,
        value=6,
        step=1
    )

    st.caption(
        "Seed menjaga agar ketiga metode menerima input noisy yang sama."
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-label">PARAMETER WIENER STFT</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="fixed-method">
        <strong>Wiener Filter (STFT)</strong>
        <span>Frame STFT: 512 sampel</span>
        <span>Alpha: 0,98</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">PARAMETER WAVELET</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="fixed-method">
        <strong>Wavelet Denoising</strong>
        <span>Wavelet family: db4</span>
        <span>Decomposition level: 3</span>
        <span>Threshold scale: 1,00</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">PARAMETER SAVITZKY-GOLAY</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="fixed-method">
        <strong>Savitzky-Golay Filter</strong>
        <span>Window length: 31</span>
        <span>Polynomial order: 3</span>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 13. PROSES UTAMA APLIKASI
# =========================================================

try:

    # -----------------------------------------------------
    # MEMBENTUK SINYAL BERSIH
    # -----------------------------------------------------

    if source == "Unggah WAV":

        if uploaded is None:
            st.info(
                "Silakan unggah file WAV melalui sidebar untuk memulai."
            )
            st.stop()

        fs, raw = wavfile.read(
            BytesIO(uploaded.getvalue())
        )

        if fs <= 0:
            raise ValueError(
                "Sampling rate WAV tidak valid."
            )

        clean_signal = to_float_audio(raw)

        # Batasi durasi maksimal 30 detik.
        clean_signal = clean_signal[:int(fs * 30)]

        sample_rate = int(fs)

        duration = len(clean_signal) / sample_rate

        clean_signal = normalize_signal(clean_signal)

    elif source == "Contoh audio referensi":

        clean_signal = make_sample_audio(
            sample_kind,
            sample_rate,
            duration
        )

    else:

        clean_signal = make_test_signal(
            sample_rate,
            duration,
            frequency
        )

    if len(clean_signal) < 100:
        raise ValueError(
            "Sinyal terlalu pendek untuk diproses."
        )

    # -----------------------------------------------------
    # MEMBUAT INPUT NOISY YANG SAMA
    # -----------------------------------------------------

    noisy_signal = add_awgn(
        clean_signal,
        noise_percent,
        int(random_seed)
    )

    # -----------------------------------------------------
    # PROSES KETIGA FILTER
    # -----------------------------------------------------

    with st.spinner(
        "Memproses Wiener, Savitzky-Golay, dan Wavelet..."
    ):

        results = {
            "Wiener Filter (STFT)": apply_wiener(
                noisy_signal,
                sample_rate
            ),

            "Savitzky-Golay Filter": apply_savgol(
                noisy_signal
            ),

            "Wavelet Denoising": apply_wavelet(
                noisy_signal
            )
        }

    # -----------------------------------------------------
    # MENGHITUNG METRIK KETIGA METODE
    # -----------------------------------------------------

    metric_rows = []

    for method, output in results.items():

        metrics = calculate_metrics(
            clean_signal,
            noisy_signal,
            output
        )

        metric_rows.append({
            "Metode": method,
            **metrics
        })

    full_df = pd.DataFrame(metric_rows)

    # Pastikan urutan metode selalu konsisten.
    full_df["Metode"] = pd.Categorical(
        full_df["Metode"],
        categories=METHOD_ORDER,
        ordered=True
    )

    full_df = (
        full_df.sort_values("Metode")
        .reset_index(drop=True)
    )

    # =====================================================
    # 14. HEADER
    # =====================================================

    st.markdown("""
    <div class="topbar">
        🎧 NoiseClean
        <span style="float:right;color:#667085;font-size:12px;">
            PSD Lab · Kelompok 6
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="hero-label">DIGITAL SIGNAL PROCESSING LAB</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-title">Analisis Perbandingan Reduksi Derau Sinyal Audio</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-description">'
        'Perbandingan Wiener Filter, Savitzky-Golay, dan Wavelet Denoising '
        'berdasarkan SNR dan RMSE.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"Eksperimen aktif | {source} | "
        f"{sample_rate:,} Hz | {duration:.2f} detik | "
        f"Seed {random_seed} | Noise {noise_percent}% RMS"
    )

    # =====================================================
    # 15. ANALISIS FOKUS SATU FILTER
    # =====================================================

    st.markdown("---")
    st.markdown("## Analisis filter")

    selected_method = st.selectbox(
        "Pilih metode yang ingin dianalisis",
        METHOD_ORDER
    )

    selected_output = results[selected_method]

    selected_metrics = full_df[
        full_df["Metode"] == selected_method
    ].iloc[0]

    st.markdown(f"### {selected_method}")

    metric_cols = st.columns(3)

    with metric_cols[0]:
        st.metric(
            "SNR (dB)",
            f"{selected_metrics['SNR (dB)']:.2f} dB"
        )

    with metric_cols[1]:
        st.metric(
            "RMSE",
            f"{selected_metrics['RMSE']:.6f}"
        )

    with metric_cols[2]:
        st.metric(
            "Peningkatan SNR",
            f"{selected_metrics['Peningkatan SNR (dB)']:+.2f} dB"
        )

    time = np.arange(len(clean_signal)) / sample_rate

    # Grafik sinyal filter yang dipilih.
    st.plotly_chart(
        make_signal_chart(
            time,
            clean_signal,
            noisy_signal,
            selected_output,
            f"Visualisasi {selected_method}"
        ),
        use_container_width=True,
        config={"displaylogo": False}
    )

    # Grafik metrik khusus metode terpilih.
    st.markdown("### Visualisasi metrik filter")

    graph_cols = st.columns(3)

    with graph_cols[0]:
        st.plotly_chart(
            make_metric_chart(
                selected_method,
                selected_metrics["SNR (dB)"],
                "SNR (dB)",
                " dB"
            ),
            use_container_width=True,
            config={"displaylogo": False}
        )

    with graph_cols[1]:
        st.plotly_chart(
            make_metric_chart(
                selected_method,
                selected_metrics["RMSE"],
                "RMSE",
                ""
            ),
            use_container_width=True,
            config={"displaylogo": False}
        )

    with graph_cols[2]:
        st.plotly_chart(
            make_metric_chart(
                selected_method,
                selected_metrics["Peningkatan SNR (dB)"],
                "Peningkatan SNR",
                " dB"
            ),
            use_container_width=True,
            config={"displaylogo": False}
        )

    # =====================================================
    # 16. PERBANDINGAN AKHIR KETIGA METODE
    # =====================================================

    st.markdown("---")
    st.markdown("## Perbandingan akhir ketiga metode")

    st.caption(
        f"Ketiga metode diuji menggunakan sinyal, sampling rate, "
        f"durasi, seed, dan noise {noise_percent}% RMS yang sama."
    )

    if len(full_df) != 3:

        st.error(
            "Data perbandingan belum lengkap. "
            "Pastikan ketiga metode berhasil diproses."
        )

    else:

        # -------------------------------------------------
        # TIGA KARTU HASIL
        # -------------------------------------------------

        st.markdown("### Ringkasan hasil setiap metode")

        cols = st.columns(3)

        for col, (_, row) in zip(
            cols,
            full_df.iterrows()
        ):

            with col:

                with st.container(border=True):

                    st.markdown(
                        f"#### {row['Metode']}"
                    )

                    st.metric(
                        "SNR",
                        f"{row['SNR (dB)']:.2f} dB"
                    )

                    st.metric(
                        "RMSE",
                        f"{row['RMSE']:.6f}"
                    )

                    st.metric(
                        "Peningkatan SNR",
                        f"{row['Peningkatan SNR (dB)']:+.2f} dB"
                    )

        # -------------------------------------------------
        # EVALUASI HASIL
        # -------------------------------------------------

        best_snr = full_df.loc[
            full_df["SNR (dB)"].idxmax()
        ]

        best_rmse = full_df.loc[
            full_df["RMSE"].idxmin()
        ]

        st.markdown("### Hasil evaluasi")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**SNR tertinggi**")

            st.success(
                f"{best_snr['Metode']} — "
                f"{best_snr['SNR (dB)']:.2f} dB"
            )

        with col2:

            st.markdown("**RMSE terendah**")

            st.success(
                f"{best_rmse['Metode']} — "
                f"{best_rmse['RMSE']:.6f}"
            )

        # -------------------------------------------------
        # GRAFIK PERBANDINGAN
        # -------------------------------------------------

        st.markdown("### Visualisasi perbandingan")

        tab1, tab2, tab3 = st.tabs([
            "Perbandingan SNR",
            "Perbandingan RMSE",
            "Peningkatan SNR"
        ])

        with tab1:

            fig = make_comparison_chart(
                full_df,
                "SNR (dB)",
                "Perbandingan SNR Ketiga Metode",
                "SNR (dB)",
                suffix=" dB",
                decimals=2
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False}
            )

        with tab2:

            fig = make_comparison_chart(
                full_df,
                "RMSE",
                "Perbandingan RMSE Ketiga Metode",
                "RMSE (lebih rendah lebih baik)",
                decimals=6
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False}
            )

        with tab3:

            fig = make_comparison_chart(
                full_df,
                "Peningkatan SNR (dB)",
                "Peningkatan SNR Ketiga Metode",
                "Peningkatan SNR (dB)",
                suffix=" dB",
                decimals=2,
                zero_line=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False}
            )

        # -------------------------------------------------
        # TABEL KOMPARASI
        # -------------------------------------------------

        st.markdown("### Tabel komparasi akhir")

        display_df = full_df.copy()

        display_df["Metode"] = (
            display_df["Metode"].astype(str)
        )

        st.dataframe(
            display_df.style.format({
                "SNR input (dB)": "{:.2f}",
                "SNR (dB)": "{:.2f}",
                "RMSE": "{:.6f}",
                "Peningkatan SNR (dB)": "{:+.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # DOWNLOAD CSV
        # -------------------------------------------------

        st.download_button(
            "⬇️ Unduh hasil perbandingan CSV",
            display_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                f"NoiseClean_Perbandingan_"
                f"{noise_percent}persen.csv"
            ),
            mime="text/csv"
        )

        st.caption(
            "SNR yang lebih tinggi menunjukkan kualitas sinyal hasil "
            "filter yang lebih baik, sedangkan RMSE yang lebih rendah "
            "menunjukkan hasil lebih dekat dengan sinyal referensi bersih."
        )

    # =====================================================
    # 17. METODOLOGI
    # =====================================================

    with st.expander("Metode dan interpretasi"):

        st.markdown("#### Wiener Filter berbasis STFT")

        st.write(
            "Wiener Filter STFT melakukan estimasi komponen noise "
            "pada domain waktu-frekuensi menggunakan Short-Time "
            "Fourier Transform (STFT), kemudian menerapkan gain "
            "untuk menekan komponen derau."
        )

        st.markdown("#### Savitzky-Golay Filter")

        st.write(
            "Savitzky-Golay melakukan pemulusan sinyal menggunakan "
            "fitting polinomial lokal pada jendela sampel."
        )

        st.markdown("#### Wavelet Denoising")

        st.write(
            "Wavelet Denoising menggunakan Discrete Wavelet Transform "
            "(DWT). Koefisien detail diberi soft threshold berdasarkan "
            "estimasi noise MAD, kemudian direkonstruksi."
        )

        st.markdown("#### Metrik evaluasi")

        st.write(
            "SNR dihitung dengan membandingkan daya sinyal referensi "
            "bersih terhadap daya error. Nilai SNR lebih tinggi "
            "menunjukkan hasil yang lebih baik."
        )

        st.write(
            "RMSE merupakan akar rata-rata kuadrat selisih antara "
            "sinyal referensi bersih dan sinyal hasil filter. "
            "Nilai RMSE lebih rendah menunjukkan hasil lebih dekat "
            "dengan sinyal referensi."
        )

        st.info(
            "Ketiga metode menggunakan sinyal noisy, sampling rate, "
            "durasi, dan seed yang sama agar perbandingan lebih adil."
        )

    # =====================================================
    # 18. FOOTER
    # =====================================================

    st.markdown("""
    <div class="footer">
        NoiseClean · Aplikasi Pengolahan Sinyal Digital · Kelompok 6
    </div>
    """, unsafe_allow_html=True)


except Exception as error:

    st.error(
        f"Tidak dapat memproses sinyal: {error}"
    )

    st.exception(error)
