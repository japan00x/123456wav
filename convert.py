import librosa
import numpy as np
import soundfile as sf
import pretty_midi
import sys
import os

def loop_pad(audio, sr, target_duration):
    target_len = int(target_duration * sr)
    if len(audio) >= target_len:
        return audio[:target_len]
    repeats = (target_len // len(audio)) + 1
    return np.tile(audio, repeats)[:target_len]

def get_base_midi(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return 60.0
    return librosa.hz_to_midi(np.median(f0))

def process_from_midi(mad_audio, sr, midi_file, target_duration):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    out_audio = np.zeros(int(target_duration * sr))
    mad_base = get_base_midi(mad_audio, sr)
    
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            start_sample = int(note.start * sr)
            end_sample = int(note.end * sr)
            if start_sample >= len(out_audio):
                continue
            end_sample = min(end_sample, len(out_audio))
            
            note_dur = (end_sample - start_sample) / sr
            slice_audio = loop_pad(mad_audio, sr, note_dur)
            
            n_steps = note.pitch - mad_base
            shifted = librosa.effects.pitch_shift(y=slice_audio, sr=sr, n_steps=n_steps)
            
            velocity_scale = note.velocity / 127.0
            out_audio[start_sample:end_sample] += shifted[:end_sample-start_sample] * velocity_scale
            
    return out_audio

def process_from_wav(mad_audio, sr, orig_audio, orig_sr):
    if sr != orig_sr:
        orig_audio = librosa.resample(y=orig_audio, orig_sr=orig_sr, target_sr=sr)
    target_duration = len(orig_audio) / sr
    looped_mad = loop_pad(mad_audio, sr, target_duration)
    
    mad_base = get_base_midi(mad_audio, sr)
    
    onset_env = librosa.onset.onset_strength(y=orig_audio, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='samples')
    onsets = np.append(onsets, len(orig_audio))
    
    out_audio = np.zeros_like(looped_mad)
    
    for i in range(len(onsets) - 1):
        start = onsets[i]
        end = onsets[i+1]
        segment_orig = orig_audio[start:end]
        
        f0, _, _ = librosa.pyin(segment_orig, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_valid = f0[~np.isnan(f0)]
        
        segment_mad = looped_mad[start:end]
        
        if len(f0_valid) > 0:
            target_midi = librosa.hz_to_midi(np.median(f0_valid))
            n_steps = target_midi - mad_base
            shifted = librosa.effects.pitch_shift(y=segment_mad, sr=sr, n_steps=n_steps)
            out_audio[start:end] = shifted
        else:
            out_audio[start:end] = segment_mad
            
    return out_audio

def main():
    mad_path = 'mad.wav'
    orig_path = 'original.wav'
    midi_path = 'original.midi'
    out_path = 'mad_output.wav'

    if not os.path.exists(mad_path) or not os.path.exists(orig_path):
        sys.exit(1)

    mad_audio, sr = librosa.load(mad_path, sr=None)
    orig_audio, orig_sr = librosa.load(orig_path, sr=None)
    target_duration = len(orig_audio) / orig_sr

    if os.path.exists(midi_path):
        result = process_from_midi(mad_audio, sr, midi_path, target_duration)
    else:
        result = process_from_wav(mad_audio, sr, orig_audio, orig_sr)
        
    result = librosa.util.normalize(result)
    sf.write(out_path, result, sr)

if __name__ == '__main__':
    main()
