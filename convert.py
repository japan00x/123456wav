import os
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from pretty_midi import PrettyMIDI
def main():
    original_wav_path = "original.wav"
    mad_wav_path = "mad.wav"
    original_midi_path = "original.midi"
    output_wav_path = "mad.wav"
    y_orig, sr = librosa.load(original_wav_path, sr=None)
    dur_orig = librosa.get_duration(y=y_orig, sr=sr)
    y_mad, sr_mad = librosa.load(mad_wav_path, sr=None)
    dur_mad = librosa.get_duration(y=y_mad, sr=sr_mad)
    if sr_mad != sr:
        y_mad = librosa.resample(y=y_mad, orig_sr=sr_mad, target_sr=sr)
    if dur_mad < dur_orig:
        diff = dur_orig - dur_mad
        prefix_len = int(diff * sr)
        prefix = y_mad[:prefix_len]
        y_mad = np.concatenate((y_mad, prefix))
    elif dur_mad > dur_orig:
        trim_len = int(dur_orig * sr)
        y_mad = y_mad[:trim_len]
    frame_period = 5.0
    f0_mad, sp_mad, ap_mad = pw.wav2world(y_mad, sr, f0_floor=71.0, f0_ceil=800.0, frame_period=frame_period)
    n_frames = len(f0_mad)
    hop_time = frame_period / 1000.0
    times = np.arange(n_frames) * hop_time
    if os.path.exists(original_midi_path):
        midi_data = PrettyMIDI(original_midi_path)
        f0_converted = np.zeros(n_frames)
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                start = note.start
                end = note.end
                pitch_hz = librosa.midi_to_hz(note.pitch)
                mask = (times >= start) & (times < end)
                f0_converted[mask] = pitch_hz
    else:
        f0_orig, _ = pw.harvest(y_orig, sr, f0_floor=71.0, f0_ceil=800.0, frame_period=frame_period)
        f0_converted = f0_orig.copy()
    y_converted = pw.synthesize(f0_converted, sp_mad, ap_mad, sr)
    rms_orig = np.sqrt(np.mean(y_orig ** 2))
    rms_mad = np.sqrt(np.mean(y_converted ** 2))
    if rms_mad > 0:
        y_converted *= (rms_orig / rms_mad)
    sf.write(output_wav_path, y_converted, sr)
if __name__ == "__main__":
    main()
