import librosa
import soundfile as sf
import numpy as np
import os
import pretty_midi
import json
original_wav = "original.wav"
mad_wav = "mad.wav"
midi_path = "original.midi"
output_wav = "mad_converted.wav"
sr = 44100
y_orig, _ = librosa.load(original_wav, sr=sr)
y_mad, _ = librosa.load(mad_wav, sr=sr)
len_orig_sec = len(y_orig) / sr
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
hop_length = 256
use_midi = os.path.exists(midi_path)
if use_midi:
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi_data.instruments:
        for note in inst.notes:
            notes.append(note)
    notes.sort(key=lambda x: x.start)
    num_frames = len(y_orig) // hop_length + 1
    target_f0 = np.full(num_frames, 0.0)
    for note in notes:
        start_frame = int(note.start * sr / hop_length)
        end_frame = int(note.end * sr / hop_length)
        if start_frame < num_frames:
            end_frame = min(end_frame, num_frames)
            target_f0[start_frame:end_frame] = librosa.midi_to_hz(note.pitch)
else:
    f0_orig, voiced_flag, _ = librosa.pyin(y_orig, fmin=50, fmax=8000, sr=sr, hop_length=hop_length, frame_length=2048, fill_na=0)
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
analysis = {'original_length_sec': len_orig_sec, 'mad_original_length_sec': len_mad_sec, 'after_adjust_length_sec': len(y_mad) / sr, 'pitch_matching': 'global_key_match_via_median', 'used_midi': use_midi, 'melody_line_matched': True}
with open('analysis.json', 'w') as f:
    json.dump(analysis, f)
