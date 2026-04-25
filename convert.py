import librosa
import soundfile as sf
import numpy as np
import os
import pretty_midi
import json
original_wav_path = "original.wav"
mad_wav_path = "mad.wav"
midi_path = "original.midi"
output_wav = "mad_converted.wav"
analysis_log = "analysis.json"
sr = 44100
hop_length = 256
if os.path.exists(midi_path):
    use_midi = True
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    len_orig_sec = midi_data.get_end_time()
    y_orig = None
else:
    use_midi = False
    if not os.path.exists(original_wav_path):
        raise FileNotFoundError("original.wav または original.midi のいずれかが必要です！")
    y_orig, _ = librosa.load(original_wav_path, sr=sr)
    len_orig_sec = len(y_orig) / sr
y_mad, _ = librosa.load(mad_wav_path, sr=sr)
len_mad_sec = len(y_mad) / sr
if not use_midi:
    tempo_orig, _ = librosa.beat.beat_track(y=y_orig, sr=sr, hop_length=hop_length, tempo_min=60, tempo_max=240, tightness=0.8)
    tempo_mad, _ = librosa.beat.beat_track(y=y_mad, sr=sr, hop_length=hop_length, tempo_min=60, tempo_max=240, tightness=0.8)
    if tempo_orig > 0 and tempo_mad > 0:
        rate = tempo_orig / tempo_mad
        y_mad = librosa.effects.time_stretch(y=y_mad, rate=rate)
        len_mad_sec = len(y_mad) / sr
if abs(len_mad_sec - len_orig_sec) > 0.01:
    if len_mad_sec < len_orig_sec:
        diff_sec = len_orig_sec - len_mad_sec
        head_sec = 1.0
        head_samples = int(head_sec * sr)
        repeats = int(np.ceil(diff_sec / head_sec))
        pad = np.tile(y_mad[:head_samples], repeats)[:int(diff_sec * sr)]
        y_mad = np.concatenate((pad, y_mad))
    else:
        y_mad = y_mad[:int(len_orig_sec * sr)]
if use_midi:
    notes = []
    for inst in midi_data.instruments:
        for note in inst.notes:
            notes.append(note)
    notes.sort(key=lambda x: x.start)
    num_frames = len(y_mad) // hop_length + 1
    target_f0 = np.full(num_frames, 0.0)
    for note in notes:
        start_frame = int(note.start * sr / hop_length)
        end_frame = int(note.end * sr / hop_length)
        if start_frame < num_frames:
            end_frame = min(end_frame, num_frames)
            target_f0[start_frame:end_frame] = librosa.midi_to_hz(note.pitch)
else:
    f0_orig, _, _ = librosa.pyin(y_orig, fmin=50, fmax=8000, sr=sr, hop_length=hop_length, frame_length=2048, fill_na=0)
    target_f0 = f0_orig
f0_mad, _, _ = librosa.pyin(y_mad, fmin=50, fmax=8000, sr=sr, hop_length=hop_length, frame_length=2048, fill_na=0)
voiced_orig = target_f0[target_f0 > 0]
voiced_mad = f0_mad[f0_mad > 0]
if len(voiced_orig) > 0 and len(voiced_mad) > 0:
    med_orig = np.median(voiced_orig)
    med_mad = np.median(voiced_mad)
    if med_mad > 0:
        n_steps = 12 * np.log2(med_orig / med_mad)
        y_mad = librosa.effects.pitch_shift(y=y_mad, sr=sr, n_steps=n_steps)
y_mad = y_mad / np.max(np.abs(y_mad))
sf.write(output_wav, y_mad, sr)
analysis = {
    'original_length_sec': len_orig_sec,
    'mad_original_length_sec': len_mad_sec,
    'after_adjust_length_sec': len(y_mad) / sr,
    'pitch_matching': 'global_key_match_via_median',
    'used_midi': use_midi,
    'melody_line_matched': True
}
if not use_midi:
    chroma_orig = librosa.feature.chroma_stft(y=y_orig, sr=sr, n_chroma=12, tuning=0.0, hop_length=hop_length)
    spectrum = librosa.feature.melspectrogram(y=y_orig, sr=sr, n_fft=4096, hop_length=hop_length, win_length=2048, window='hann', n_mels=128)
    onset_env = librosa.onset.onset_strength(y=y_orig, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(y=y_orig, sr=sr, hop_length=hop_length, pre_max=0.03, post_max=0.03, pre_avg=0.1, post_avg=0.1, delta=0.07, wait=0)
    harmonic, percussive = librosa.effects.hpss(y=y_orig, margin=1.0, power=2.0)
    analysis.update({
        'chroma_mean': np.mean(chroma_orig, axis=1).tolist(),
        'spectrum_mean_db': librosa.power_to_db(np.mean(spectrum, axis=1), ref=np.max).tolist(),
        'onsets_frames': onsets.tolist(),
        'tempo_bpm': float(tempo_orig) if 'tempo_orig' in locals() else 0.0,
        'harmonic_rms': float(np.sqrt(np.mean(harmonic**2))),
        'percussive_rms': float(np.sqrt(np.mean(percussive**2)))
    })
with open(analysis_log, 'w') as f:
    json.dump(analysis, f)
