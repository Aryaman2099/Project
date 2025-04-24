import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from librosa.sequence import dtw

# ------------- LOAD CHORD DATA FROM JSON ---------------- #
def load_chords_from_dataset(json_path, song_index):
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    song_data = dataset[song_index]
    chords = [entry['chord'] for entry in song_data['chords']]
    return chords, song_data['audio_file']

# ------------- CHORD TO VECTOR ---------------- #
def chord_to_vector(chord, pitch_classes):
    vec = np.zeros(12)
    # Handle sharp chords like C#, G#m etc.
    root = chord[:2] if chord[:2] in pitch_classes else chord[0]
    idx = pitch_classes.index(root)
    
    if 'm' in chord and 'maj' not in chord:
        intervals = [0, 3, 7]
    elif '7' in chord:
        intervals = [0, 4, 7, 10]
    else:
        intervals = [0, 4, 7]
    
    for i in intervals:
        vec[(idx + i) % 12] = 1
    return vec

# ------------- SIMILARITY COMPUTATION ---------------- #
def compute_similarity(chords1, chords2, pitch_classes):
    vecs1 = np.array([chord_to_vector(ch, pitch_classes) for ch in chords1])
    vecs2 = np.array([chord_to_vector(ch, pitch_classes) for ch in chords2])
    
    cost_matrix = cdist(vecs1, vecs2, metric='cosine')
    D, wp = dtw(C=cost_matrix)  # <-- FIXED HERE
    sim_score = 1 - D[-1, -1] / len(wp)
    return D, wp, sim_score


# ------------- PLOT ALIGNMENT ---------------- #
def plot_alignment(chords1, chords2, wp):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Chord Alignment (DTW Path)", fontsize=14)
    for i, j in wp:
        ax.plot([0, 1], [i, j], color='skyblue', linewidth=0.8)
    ax.set_yticks(np.arange(len(chords1)))
    ax.set_yticklabels(chords1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Song 1", "Song 2"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(len(chords1), len(chords2)))
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

# ------------- MAIN COMPARISON FUNCTION ---------------- #
def compare_songs_from_dataset(json_path, index1, index2):
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',
                     'G', 'G#', 'A', 'A#', 'B']

    chords1, name1 = load_chords_from_dataset(json_path, index1)
    chords2, name2 = load_chords_from_dataset(json_path, index2)

    print(f"\n🔍 Comparing: {name1} vs {name2}")
    print(f"🎵 Chord Count → {len(chords1)} vs {len(chords2)}")

    D, wp, score = compute_similarity(chords1, chords2, pitch_classes)
    print(f"✅ Similarity Score: {score:.3f}")

    plot_alignment(chords1, chords2, wp)
    return score

# ------------- ENTRY POINT ---------------- #
if __name__ == "__main__":
    # Path to the uploaded dataset
    json_path = r"C:\Users\aryam\OneDrive\Desktop\Artificial Intelligence\lab\Project\output_midi\all_chords_dataset.json"

    # Choose indices of the songs to compare
    index1 = 0  # first song
    index2 = 5 # second song (change as needed)

    compare_songs_from_dataset(json_path, index1, index2)
