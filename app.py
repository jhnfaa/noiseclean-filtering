
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pywt
import streamlit as st

from scipy.io import wavfile
from scipy.signal import stft, istft, savgol_filter


# =========================================================
# KONFIGURASI APLIKASI
# =========================================================

st.set_page_config(
    page_title="NoiseClean | PSD",
    page_icon="🎧",
    layout="wide"
)

TARGET_SNR_DB = 35.0

# Parameter eksperimen tetap
WIENER_FRAME = 512
WIENER_ALPHA = 0.98

SAVGOL_WINDOW = 31
SAVGOL_POLYORDER = 3

WAVELET_NAME = "db4"
WAVELET_LEVEL = 3
WAVELET_THRESHOLD = 1.0


# =========================================================
# CSS
# =========================================================

def inject_css():
    st.markdown("""
    <style>
    .stApp {
        background: #f3f5fb;
        color: #172033;
    }

    [data-testid="stSidebar"] {
        background: #151d32;
    }

    [data-testid="stSidebar"] * {
        color: #e8ecf7;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #aab5cc;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e7f0;
        border-radius: 13px;
        padding: 18px;
        min-height: 120px;
    }

    [data-testid="stPlotlyChart"] {
        background: white;
        border: 1px solid #e2e7f0;
        border-radius: 12px;
        padding: 8px;
    }

    .topbar {
        background: white;
        padding: 15px 20px;
        border: 1px solid #e2e7f0;
        border-radius: 13px;
        margin-bottom: 25px;
        font-weight: 800;
    }

    .hero-label {
        color: #5b5ce2;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 850;
        color: #172033;
        margin-top: 8px;
    }

    .hero-description {
        color: #667085;
        margin-bottom: 20px;
    }

    .section-label {
        color: #667085;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        color: #98a2b3;
        padding: 30px 0;
        font-size: 12px;
    }

    </style>
    """, unsafe_allow_html=True)


# =========================================================
# FUNGSI SINYAL
# =========================================================

def to_float_audio(samples):
    raw = np.asarray(samples)

    if raw.ndim == 2:
        raw = raw.mean(axis=1)

    if raw.dtype == np.uint8:
        values = (raw.astype(float) - 128) / 128

    elif np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        scale = max(abs(info.min), info.max)
        values = raw.astype(float) / scale

    else:
        values = raw.astype(float)

    values = np.nan_to_num(values)

    return values


def normalize_signal(x):
    x = np.asarray(x, dtype=float)

    peak = np.max(np.abs(x))

    if peak == 0:
        raise ValueError("Sinyal kosong atau memiliki amplitudo nol.")

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
            (duration - t) / 0.1, 1
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
    Menambahkan Gaussian noise berdasarkan persentase RMS
    sinyal bersih.
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
# WIENER FILTER STFT
# =========================================================

def apply_wiener(signal, fs):
    x = np.asarray(signal, dtype=float)

    nperseg = min(WIENER_FRAME, len(x))

    if nperseg < 16:
        raise ValueError("Sinyal terlalu pendek untuk Wiener STFT.")

    noverlap = nperseg // 2

    frequencies, times, Z = stft(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros"
    )

    power = np.abs(Z) ** 2

    # Estimasi noise berdasarkan persentil daya antar-frame.
    noise_power = np.percentile(
        power,
        20,
        axis=1,
        keepdims=True
    )

    # Gain Wiener dengan alpha sebagai faktor penghalus.
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
# SAVITZKY-GOLAY
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
# WAVELET DENOISING
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
# METRIK EVALUASI
# =========================================================

def calculate_metrics(clean, noisy, filtered):
    clean = np.asarray(clean)
    noisy = np.asarray(noisy)
    filtered = np.asarray(filtered)

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
# GRAFIK
# =========================================================

def make_signal_chart(time, clean, noisy, filtered, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time,
        y=clean,
        name="Sinyal bersih",
        line=dict(color="#16a085", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=time,
        y=noisy,
        name="Sinyal noisy",
        line=dict(color="#e67e22", width=1)
    ))

    fig.add_trace(go.Scatter(
        x=time,
        y=filtered,
        name="Sinyal hasil filter",
        line=dict(color="#4f46e5", width=1.5)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Waktu (detik)",
        yaxis_title="Amplitudo",
        height=420,
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.12)
    )

    return fig


def make_metric_chart(name, value, title, suffix=""):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[name],
        y=[value],
        text=[f"{value:.4f}{suffix}"],
        textposition="outside",
        marker_color="#5b5ce2"
    ))

    fig.update_layout(
        title=title,
        height=320,
        showlegend=False,
        yaxis_title=title,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    return fig


# =========================================================
# SIDEBAR
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
            1.0,
            10.0,
            5.0,
            0.5
        )

        frequency = st.slider(
            "Frekuensi fundamental (Hz)",
            100,
            min(2000, sample_rate // 2 - 1),
            min(440, sample_rate // 2 - 1),
            10
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
# MEMBENTUK SINYAL BERSIH
# =========================================================

try:

    if source == "Unggah WAV":

        if uploaded is None:
            st.info(
                "Silakan unggah file WAV melalui sidebar untuk memulai."
            )
            st.stop()

        fs, raw = wavfile.read(
            BytesIO(uploaded.getvalue())
        )

        clean_signal = to_float_audio(raw)

        if fs <= 0:
            raise ValueError("Sampling rate WAV tidak valid.")

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
        raise ValueError("Sinyal terlalu pendek untuk diproses.")

    # Input noisy yang sama untuk seluruh metode.
    noisy_signal = add_awgn(
        clean_signal,
        noise_percent,
        int(random_seed)
    )

    # =====================================================
    # PROSES KETIGA FILTER
    # =====================================================

    with st.spinner("Memproses ketiga metode filtering..."):

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

    # =====================================================
    # HEADER
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
    # TAMPILAN FOKUS SATU FILTER
    # =====================================================

    st.markdown("## Analisis filter")

    selected_method = st.selectbox(
        "Pilih metode yang ingin dianalisis",
        list(results.keys())
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

    st.plotly_chart(
        make_signal_chart(
            time,
            clean_signal,
            noisy_signal,
            selected_output,
            f"Visualisasi {selected_method}"
        ),
        use_container_width=True
    )

    # Grafik metrik khusus metode yang dipilih.
    graph_cols = st.columns(3)

    with graph_cols[0]:
        st.plotly_chart(
            make_metric_chart(
                selected_method,
                selected_metrics["SNR (dB)"],
                "SNR (dB)",
                " dB"
            ),
            use_container_width=True
        )

    with graph_cols[1]:
        st.plotly_chart(
            make_metric_chart(
                selected_method,
                selected_metrics["RMSE"],
                "RMSE"
            ),
            use_container_width=True
        )

    with graph_cols[2]:
        st.plotly_chart(
            make_metric_chart(
                selected_method,
                selected_metrics["Peningkatan SNR (dB)"],
                "Peningkatan SNR",
                " dB"
            ),
            use_container_width=True
        )

    # =====================================================
    # PERBANDINGAN AKHIR KETIGA METODE
    # =====================================================

    st.markdown("---")

    st.markdown("## Perbandingan akhir ketiga metode")

    st.caption(
        f"Semua metode diuji menggunakan sinyal, sampling rate, "
        f"durasi, seed, dan noise {noise_percent}% RMS yang sama."
    )

    comparison = full_df.sort_values(
        "SNR (dB)",
        ascending=False
    ).reset_index(drop=True)

    best_snr = comparison.iloc[0]

    best_rmse = full_df.loc[
        full_df["RMSE"].idxmin()
    ]

    summary_cols = st.columns(2)

    with summary_cols[0]:
        st.metric(
            "SNR tertinggi",
            str(best_snr["Metode"]),
            f"{best_snr['SNR (dB)']:.2f} dB"
        )

    with summary_cols[1]:
        st.metric(
            "RMSE terendah",
            str(best_rmse["Metode"]),
            f"{best_rmse['RMSE']:.6f}"
        )

    # Kartu masing-masing metode.
    st.markdown("### Hasil setiap metode")

    summary_cards = st.columns(3)

    for col, (_, row) in zip(
        summary_cards,
        full_df.iterrows()
    ):

        with col:

            st.markdown(f"#### {row['Metode']}")

            st.metric(
                "SNR (dB)",
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

    compare_tabs = st.tabs([
        "Grafik SNR",
        "Grafik RMSE",
        "Peningkatan SNR"
    ])

    colors = [
        "#4f46e5",
        "#0f9f8a",
        "#f59e0b"
    ]

    with compare_tabs[0]:

        fig = go.Figure(go.Bar(
            x=full_df["Metode"],
            y=full_df["SNR (dB)"],
            text=[
                f"{v:.2f} dB"
                for v in full_df["SNR (dB)"]
            ],
            textposition="outside",
            marker_color=colors
        ))

        fig.update_layout(
            title="Perbandingan SNR ketiga metode",
            xaxis_title="Metode",
            yaxis_title="SNR (dB)",
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with compare_tabs[1]:

        fig = go.Figure(go.Bar(
            x=full_df["Metode"],
            y=full_df["RMSE"],
            text=[
                f"{v:.6f}"
                for v in full_df["RMSE"]
            ],
            textposition="outside",
            marker_color=colors
        ))

        fig.update_layout(
            title="Perbandingan RMSE ketiga metode",
            xaxis_title="Metode",
            yaxis_title="RMSE (lebih rendah lebih baik)",
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with compare_tabs[2]:

        fig = go.Figure(go.Bar(
            x=full_df["Metode"],
            y=full_df["Peningkatan SNR (dB)"],
            text=[
                f"{v:+.2f} dB"
                for v in full_df["Peningkatan SNR (dB)"]
            ],
            textposition="outside",
            marker_color=colors
        ))

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#64748b"
        )

        fig.update_layout(
            title="Peningkatan SNR terhadap sinyal noisy",
            xaxis_title="Metode",
            yaxis_title="Peningkatan SNR (dB)",
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # TABEL KOMPARASI
    # =====================================================

    st.markdown("### Tabel komparasi")

    st.dataframe(
        comparison.style.format({
            "SNR input (dB)": "{:.2f}",
            "SNR (dB)": "{:.2f}",
            "RMSE": "{:.6f}",
            "Peningkatan SNR (dB)": "{:+.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Unduh perbandingan ketiga metode (CSV)",
        comparison.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=f"noiseclean_comparison_noise_{noise_percent}pct.csv",
        mime="text/csv"
    )

    # =====================================================
    # INFORMASI PARAMETER
    # =====================================================

    with st.expander("Metode dan parameter eksperimen"):

        st.markdown("#### Wiener Filter (STFT)")

        st.write(
            f"Frame STFT: {WIENER_FRAME} sampel"
        )

        st.write(
            f"Alpha: {WIENER_ALPHA}"
        )

        st.markdown("#### Savitzky-Golay Filter")

        st.write(
            f"Window length: {SAVGOL_WINDOW}"
        )

        st.write(
            f"Polynomial order: {SAVGOL_POLYORDER}"
        )

        st.markdown("#### Wavelet Denoising")

        st.write(
            f"Wavelet family: {WAVELET_NAME}"
        )

        st.write(
            f"Decomposition level: {WAVELET_LEVEL}"
        )

        st.write(
            f"Threshold scale: {WAVELET_THRESHOLD}"
        )

        st.markdown("#### Metrik evaluasi")

        st.write(
            "SNR dihitung terhadap clean reference. "
            "Nilai lebih tinggi menunjukkan hasil reduksi noise lebih baik."
        )

        st.write(
            "RMSE merupakan akar rata-rata kuadrat error "
            "antara sinyal hasil filtering dan clean reference. "
            "Nilai lebih rendah menunjukkan error lebih kecil."
        )

        st.warning(
            "Hasil pada sinyal sintetis merupakan pengujian simulasi. "
            "Validasi pada rekaman audio asli tetap diperlukan."
        )

    st.markdown(
        '<div class="footer">'
        'NoiseClean · Aplikasi Pengolahan Sinyal Digital · Kelompok 6'
        '</div>',
        unsafe_allow_html=True
    )

except Exception as error:

    st.error(
        f"Tidak dapat memproses sinyal: {error}"
    )

    st.exception(error)
