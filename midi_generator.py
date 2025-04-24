import os
import json
import subprocess
import librosa
import pretty_midi
import numpy as np
from glob import glob
from tqdm import tqdm

# Set FFmpeg path
ffmpeg_path = r"C:\Users\aryam\anaconda3\Library\bin\ffmpeg.exe"

# Input & output paths
input_dir = r"C:\Users\aryam\OneDrive\Desktop\Artificial Intelligence\lab\Project\songs"
output_dir = r"C:\Users\aryam\OneDrive\Desktop\Artificial Intelligence\lab\Project\output_midi"
os.makedirs(output_dir, exist_ok=True)

# Convert .mp3 and .mp4 to .wav
def convert_to_wav(input_path, output_path):
    command = [
        ffmpeg_path, '-y', '-i', input_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Converted {os.path.basename(input_path)} to WAV")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to convert {input_path}")

# Chord converter
class AudioToMidiChordConverter:
    def __init__(self, output_dir, sample_rate=22050, hop_length=512):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        os.makedirs(self.output_dir, exist_ok=True)

        self.chord_to_midi = {
            'C': [60, 64, 67],
            'Cm': [60, 63, 67],
            'C7': [60, 64, 67, 70],
            'Cmaj7': [60, 64, 67, 71],
            'Cm7': [60, 63, 67, 70],
        }

        self.pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',
                              'G', 'G#', 'A', 'A#', 'B']

    def detect_chords(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        y_harmonic = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=self.hop_length)
        timestamps = librosa.times_like(chroma, sr=sr, hop_length=self.hop_length)

        chord_sequence = []
        current_chord = None
        start_time = 0

        for i, frame in enumerate(chroma.T):
            if timestamps[i] > 80.0:
                break

            max_idx = np.argmax(frame)
            chord_root = self.pitch_classes[max_idx]
            minor_third = frame[(max_idx + 3) % 12]
            major_third = frame[(max_idx + 4) % 12]
            seventh = frame[(max_idx + 10) % 12]

            if major_third > minor_third and seventh > 0.5:
                chord = f"{chord_root}7"
            elif major_third > minor_third:
                chord = chord_root
            else:
                chord = f"{chord_root}m"

            if chord != current_chord and i > 0:
                if current_chord:
                    chord_sequence.append((current_chord, start_time, timestamps[i - 1]))
                current_chord = chord
                start_time = timestamps[i - 1]

        if current_chord and start_time < 60.0:
            end_time = min(timestamps[-1], 60.0)
            chord_sequence.append((current_chord, start_time, end_time))

        return chord_sequence

    def convert_chords_to_midi(self, chord_sequence, output_path):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)

        for chord_name, start_time, end_time in chord_sequence:
            basic_chord = chord_name.split('/')[0]

            if basic_chord not in self.chord_to_midi:
                try:
                    root = basic_chord[:2] if basic_chord[:2] in self.pitch_classes else basic_chord[0]
                    root_idx = self.pitch_classes.index(root)
                    if 'm' in basic_chord and 'maj' not in basic_chord:
                        notes = [60 + root_idx, 60 + root_idx + 3, 60 + root_idx + 7]
                    else:
                        notes = [60 + root_idx, 60 + root_idx + 4, 60 + root_idx + 7]
                except:
                    continue
            else:
                notes = self.chord_to_midi[basic_chord]

            for note_number in notes:
                note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=end_time)
                piano.notes.append(note)

        midi.instruments.append(piano)
        midi.write(output_path)

    def process(self, audio_path):
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        midi_path = os.path.join(self.output_dir, f"{filename}.mid")
        chord_sequence = self.detect_chords(audio_path)
        self.convert_chords_to_midi(chord_sequence, midi_path)

        return {
            'audio_file': os.path.basename(audio_path),
            'midi_file': os.path.basename(midi_path),
            'chords': [{'chord': c, 'start': s, 'end': e} for c, s, e in chord_sequence]
        }

# Main run
# Step 1: Convert MP3/MP4 to WAV
media_files = glob(os.path.join(input_dir, "*.mp3")) + glob(os.path.join(input_dir, "*.mp4"))
for file in media_files:
    wav_out = os.path.splitext(file)[0] + ".wav"
    convert_to_wav(file, wav_out)

# Step 2: Process all WAVs and collect dataset
converter = AudioToMidiChordConverter(output_dir)
wav_files = glob(os.path.join(input_dir, "*.wav"))
print(f"🎧 Found {len(wav_files)} WAV files to process.")

full_dataset = []
for wav in tqdm(wav_files):
    try:
        entry = converter.process(wav)
        full_dataset.append(entry)
    except Exception as e:
        print(f"❌ Error processing {wav}: {e}")

# Step 3: Save all chords to single JSON file
json_path = os.path.join(output_dir, "all_chords_dataset.json")
with open(json_path, "w") as f:
    json.dump(full_dataset, f, indent=2)

print(f"📁 All chord data saved to: {json_path}")
