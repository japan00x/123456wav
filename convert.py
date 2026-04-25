import sys
import warnings
import numpy as np
import librosa
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt, find_peaks
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

HOP_LENGTH = 256
N_FFT = 2048
WIN_LENGTH = 1024
FRAME_SIZE = 1024
CROSSFADE_RATIO = 0.05
PITCH_BINS_PER_OCTAVE = 96
FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C7")
SMOOTH_SIGMA = 3.0
RMS_EPSILON = 1e-9
VOICED_PROB_THRESHOLD = 0.4


def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float64), sr


def resample_to(y, sr_orig, sr_target):
    if sr_orig == sr_target:
        return y
    return librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)


def compute_stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                        window="hann", center=True, pad_mode="reflect")


def compute_spectral_features(y, sr, hop_length=HOP_LENGTH):
    S = np.abs(compute_stft(y))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=WIN_LENGTH, hop_length=hop_length)[0]
    return {
        "magnitude": S,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness,
        "contrast": contrast,
        "rms": rms,
        "zcr": zcr,
    }


def extract_pitch_pyin(y, sr, hop_length=HOP_LENGTH):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        hop_length=hop_length,
        win_length=WIN_LENGTH,
        n_thresholds=100,
        beta_parameters=(2, 18),
        boltzmann_parameter=2,
        resolution=0.1,
        max_transition_rate=35.92,
        switch_prob=0.01,
        no_trough_prob=0.01,
    )
    voiced_probs = np.nan_to_num(voiced_probs, nan=0.0)
    reliable_voiced = voiced_flag & (voiced_probs >= VOICED_PROB_THRESHOLD)
    return f0, reliable_voiced, voiced_probs


def hz_to_midi(hz_array):
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = np.where(
            hz_array > 0,
            12.0 * np.log2(np.where(hz_array > 0, hz_array, 1.0) / 440.0) + 69.0,
            np.nan,
        )
    return midi


def midi_to_hz(midi_array):
    return np.where(
        ~np.isnan(midi_array),
        440.0 * np.power(2.0, (midi_array - 69.0) / 12.0),
        np.nan,
    )


def smooth_f0(f0, voiced, sigma=SMOOTH_SIGMA):
    f0_out = f0.copy()
    valid = voiced & ~np.isnan(f0)
    if np.sum(valid) < 2:
        return f0_out
    indices = np.where(valid)[0]
    values = f0[valid]
    interp_fn = interp1d(indices, values, kind="linear", bounds_error=False,
                         fill_value=(values[0], values[-1]))
    filled = interp_fn(np.arange(len(f0)))
    smoothed = gaussian_filter1d(filled, sigma=sigma)
    f0_out = np.where(valid, smoothed, f0_out)
    return f0_out


def detect_scale(f0, voiced):
    valid_f0 = f0[voiced & ~np.isnan(f0)]
    if len(valid_f0) == 0:
        return None, np.zeros(12), None
    midi_notes = hz_to_midi(valid_f0)
    midi_notes = midi_notes[~np.isnan(midi_notes)]
    pitch_classes = np.round(midi_notes % 12).astype(int) % 12
    pc_histogram = np.bincount(pitch_classes, minlength=12).astype(float)
    pc_histogram /= (pc_histogram.sum() + 1e-9)

    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)

    best_score = -np.inf
    best_root = 0
    best_mode = "major"

    for root in range(12):
        shifted_major = np.roll(major_template, root)
        shifted_minor = np.roll(minor_template, root)
        score_major = np.dot(pc_histogram, shifted_major)
        score_minor = np.dot(pc_histogram, shifted_minor)
        if score_major > best_score:
            best_score = score_major
            best_root = root
            best_mode = "major"
        if score_minor > best_score:
            best_score = score_minor
            best_root = root
            best_mode = "minor"

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    scale_name = f"{note_names[best_root]} {best_mode}"
    median_octave = int(np.median(midi_notes) / 12) - 1
    return scale_name, pc_histogram, median_octave


def analyze_velocity_envelope(rms, sr, hop_length=HOP_LENGTH):
    rms_db = librosa.amplitude_to_db(rms + RMS_EPSILON, ref=np.max(rms + RMS_EPSILON))
    velocity = np.clip((rms_db + 80.0) / 80.0, 0.0, 1.0)
    midi_velocity = (velocity * 127.0).astype(int)
    peaks, props = find_peaks(rms, height=np.mean(rms) * 0.5, distance=int(sr * 0.05 / hop_length))
    return velocity, midi_velocity, peaks, rms_db


def analyze_tempo_and_beats(y, sr, hop_length=HOP_LENGTH):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length,
                                              aggregate=np.median, fmax=8000)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr,
                                            hop_length=hop_length, tightness=100)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    dynamic_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
                                           hop_length=hop_length, aggregate=None)
    return float(tempo), beat_times, dynamic_tempo, onset_env


def compute_chroma(y, sr, hop_length=HOP_LENGTH):
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length,
                                              bins_per_octave=PITCH_BINS_PER_OCTAVE,
                                              fmin=FMIN, norm=np.inf, threshold=0.0,
                                              n_chroma=12, n_octaves=7)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length,
                                               fmin=FMIN, n_chroma=12, n_octaves=7,
                                               bins_per_octave=PITCH_BINS_PER_OCTAVE)
    return chroma_cqt, chroma_cens


def loop_crossfade_interpolate(y, target_length, sr, crossfade_ratio=CROSSFADE_RATIO):
    src_len = len(y)
    if src_len >= target_length:
        return y[:target_length].copy()

    crossfade_len = max(1, int(src_len * crossfade_ratio))
    fade_out = np.linspace(1.0, 0.0, crossfade_len)
    fade_in = np.linspace(0.0, 1.0, crossfade_len)

    result = np.zeros(target_length, dtype=np.float64)
    pos = 0
    loop_body = y.copy()

    loop_body[-crossfade_len:] *= fade_out
    loop_body[:crossfade_len] *= fade_in

    blend_start = src_len - crossfade_len

    while pos < target_length:
        remaining = target_length - pos
        if remaining >= src_len:
            result[pos:pos + blend_start] += loop_body[:blend_start]
            if pos + blend_start + crossfade_len <= target_length:
                result[pos + blend_start:pos + blend_start + crossfade_len] += loop_body[blend_start:]
            pos += blend_start
        else:
            chunk = loop_body[:remaining]
            result[pos:pos + remaining] += chunk
            pos += remaining

    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    return result


def align_length(y_src, target_length, sr):
    src_len = len(y_src)
    if src_len == target_length:
        return y_src
    if src_len < target_length:
        return loop_crossfade_interpolate(y_src, target_length, sr)
    return y_src[:target_length]


def frame_pitch_shift_psola(y, sr, f0_src, f0_tgt, voiced_tgt,
                             hop_length=HOP_LENGTH, bins_per_octave=PITCH_BINS_PER_OCTAVE):
    n_frames = min(len(f0_src), len(f0_tgt))
    output = np.zeros(len(y), dtype=np.float64)
    weight = np.zeros(len(y), dtype=np.float64)
    hann_full = np.hanning(hop_length * 2)

    for i in range(n_frames):
        start = i * hop_length
        end = min(start + hop_length * 2, len(y))
        if start >= len(y):
            break

        src_f0 = f0_src[i] if i < len(f0_src) else np.nan
        tgt_f0 = f0_tgt[i] if i < len(f0_tgt) else np.nan
        is_voiced = voiced_tgt[i] if i < len(voiced_tgt) else False

        frame = y[start:end]
        frame_len = len(frame)
        if frame_len == 0:
            continue

        if is_voiced and not np.isnan(src_f0) and not np.isnan(tgt_f0) and src_f0 > 0 and tgt_f0 > 0:
            n_steps = 12.0 * np.log2(tgt_f0 / src_f0)
            if abs(n_steps) > 0.05:
                shifted = librosa.effects.pitch_shift(
                    frame.astype(np.float32), sr=sr, n_steps=n_steps,
                    bins_per_octave=bins_per_octave, res_type="kaiser_best"
                ).astype(np.float64)
            else:
                shifted = frame.copy()
        else:
            shifted = frame.copy()

        win = hann_full[:frame_len] if frame_len <= len(hann_full) else np.hanning(frame_len)
        write_len = min(frame_len, len(output) - start)
        output[start:start + write_len] += shifted[:write_len] * win[:write_len]
        weight[start:start + write_len] += win[:write_len]

    nonzero = weight > RMS_EPSILON
    output[nonzero] /= weight[nonzero]
    return output


def apply_rms_normalization(y_out, y_ref, hop_length=HOP_LENGTH):
    rms_ref = librosa.feature.rms(y=y_ref, frame_length=N_FFT, hop_length=hop_length)[0]
    rms_out = librosa.feature.rms(y=y_out, frame_length=N_FFT, hop_length=hop_length)[0]

    n_frames = min(len(rms_ref), len(rms_out))
    scale_factors = np.where(
        rms_out[:n_frames] > RMS_EPSILON,
        rms_ref[:n_frames] / (rms_out[:n_frames] + RMS_EPSILON),
        1.0,
    )
    scale_factors = gaussian_filter1d(scale_factors, sigma=5.0)
    scale_factors = np.clip(scale_factors, 0.1, 10.0)

    sample_indices = librosa.frames_to_samples(np.arange(n_frames), hop_length=hop_length)
    sample_indices = np.minimum(sample_indices, len(y_out) - 1)
    scale_interp = interp1d(
        sample_indices, scale_factors,
        kind="linear", bounds_error=False,
        fill_value=(scale_factors[0], scale_factors[-1])
    )
    sample_scale = scale_interp(np.arange(len(y_out)))
    return y_out * sample_scale


def butter_bandpass(y, sr, lowcut=80.0, highcut=8000.0, order=4):
    nyq = sr / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfilt(sos, y)


def process_audio(original_path, mad_path, output_path):
    y_orig, sr_orig = load_audio(original_path)
    y_mad, sr_mad = load_audio(mad_path)

    sr = max(sr_orig, sr_mad)
    if sr_orig != sr:
        y_orig = resample_to(y_orig, sr_orig, sr)
    if sr_mad != sr:
        y_mad = resample_to(y_mad, sr_mad, sr)

    orig_len = len(y_orig)
    y_mad = align_length(y_mad, orig_len, sr)

    f0_orig, voiced_orig, probs_orig = extract_pitch_pyin(y_orig, sr)
    f0_mad, voiced_mad, probs_mad = extract_pitch_pyin(y_mad, sr)

    f0_orig_smooth = smooth_f0(f0_orig, voiced_orig)
    f0_mad_smooth = smooth_f0(f0_mad, voiced_mad)

    scale_orig, pc_hist_orig, octave_orig = detect_scale(f0_orig_smooth, voiced_orig)
    scale_mad, pc_hist_mad, octave_mad = detect_scale(f0_mad_smooth, voiced_mad)

    spec_orig = compute_spectral_features(y_orig, sr)
    spec_mad = compute_spectral_features(y_mad, sr)

    tempo_orig, beats_orig, dyn_tempo_orig, onset_orig = analyze_tempo_and_beats(y_orig, sr)
    tempo_mad, beats_mad, dyn_tempo_mad, onset_mad = analyze_tempo_and_beats(y_mad, sr)

    velocity_orig, midi_vel_orig, peaks_orig, rms_db_orig = analyze_velocity_envelope(spec_orig["rms"], sr)
    velocity_mad, midi_vel_mad, peaks_mad, rms_db_mad = analyze_velocity_envelope(spec_mad["rms"], sr)

    chroma_orig, chroma_cens_orig = compute_chroma(y_orig, sr)
    chroma_mad, chroma_cens_mad = compute_chroma(y_mad, sr)

    n_frames = min(len(f0_orig_smooth), len(f0_mad_smooth))

    target_f0 = np.full(n_frames, np.nan)
    for i in range(n_frames):
        orig_v = voiced_orig[i] if i < len(voiced_orig) else False
        mad_v = voiced_mad[i] if i < len(voiced_mad) else False
        o_f0 = f0_orig_smooth[i] if i < len(f0_orig_smooth) else np.nan
        m_f0 = f0_mad_smooth[i] if i < len(f0_mad_smooth) else np.nan

        if orig_v and not np.isnan(o_f0):
            target_f0[i] = o_f0
        elif mad_v and not np.isnan(m_f0):
            target_f0[i] = m_f0

    target_voiced = ~np.isnan(target_f0)
    f0_mad_aligned = f0_mad_smooth[:n_frames]

    y_shifted = frame_pitch_shift_psola(
        y_mad, sr, f0_mad_aligned, target_f0, target_voiced
    )

    y_shifted = apply_rms_normalization(y_shifted, y_orig)

    y_shifted = butter_bandpass(y_shifted, sr, lowcut=60.0, highcut=min(18000.0, sr / 2.0 - 100))

    peak_val = np.max(np.abs(y_shifted))
    if peak_val > 0.98:
        y_shifted = y_shifted * (0.98 / peak_val)

    y_shifted = np.clip(y_shifted, -1.0, 1.0).astype(np.float32)

    sf.write(output_path, y_shifted, sr, subtype="PCM_16")

    print(f"original: {original_path}")
    print(f"  sr={sr}Hz  frames={orig_len}  duration={orig_len/sr:.3f}s")
    print(f"  scale={scale_orig}  tempo={tempo_orig:.2f}bpm")
    print(f"  voiced_ratio={np.mean(voiced_orig):.3f}")
    print(f"  rms_mean={np.mean(spec_orig['rms']):.5f}")
    print(f"mad: {mad_path}")
    print(f"  scale={scale_mad}  tempo={tempo_mad:.2f}bpm")
    print(f"  voiced_ratio={np.mean(voiced_mad):.3f}")
    print(f"output: {output_path}")
    print(f"  frames={len(y_shifted)}  duration={len(y_shifted)/sr:.3f}s")


if __name__ == "__main__":
    original_path = sys.argv[1] if len(sys.argv) > 1 else "original.wav"
    mad_path = sys.argv[2] if len(sys.argv) > 2 else "mad.wav"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output.wav"
    process_audio(original_path, mad_path, output_path)
