import json
import os
import librosa
import librosa.feature
import math

DATASET_PATH = "Data/genres_original"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    num_samples_per_segment = SAMPLES_PER_TRACK // num_segments
    expected_n_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_comp = dirpath.split("//")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing {semantic_label}")
            for f in filenames:
                # Getting Files 1 by 1
                file_path = os.path.join(dirpath, f)

                # Calculate Signal for each File
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                for s in range(num_segments):
                    # Divide each file into Segments
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    # Calculate 13 MFCC Vectors for each segment
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # Discard segments shorter than expected
                    if len(mfcc) == expected_n_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data['labels'].append(i - 1)
                        print(f"{file_path},segment:{s + 1}")
                print()

    # Put all data in JSON File
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
