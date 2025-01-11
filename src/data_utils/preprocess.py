import os

from tqdm import tqdm
import numpy as np
import pandas as pd

from config import data_settings as ds, emotion_to_index


def load_and_process_df(file_path): 
    df = pd.read_csv(file_path)
    df = (
        df.groupby(["art_style", "painting"])
        .agg({"emotion": list, "utterance": list})
        .reset_index()
    )

    df["binary_labels"] = [
        convert_labels_to_binary(labels) for labels in df["emotion"]
    ]

    df = keep_existing_paintings(df)

    return df



def convert_labels_to_binary(labels):
    binary_label = np.zeros(len(ds.emotion_list), dtype=np.float32)
    for label in labels:
        binary_label[emotion_to_index[label]] = 1
    return binary_label


def keep_existing_paintings(df):
    missing_images = []
    valid_indices = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        art_style = row["art_style"]
        painting = row["painting"]
        image_path = f"{ds.wikiart_dir}/{art_style}/{painting}.jpg"

        if os.path.exists(image_path):
            valid_indices.append(idx)
        else:
            missing_images.append(
                {"index": idx, "style": art_style, "painting": painting, "path": image_path}
            )

    clean_df = df.loc[valid_indices].reset_index(drop=True)

    return clean_df

