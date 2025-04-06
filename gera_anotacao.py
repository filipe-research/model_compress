# 1.	Lê o annotations.txt.
# 2.	Filtra apenas as pessoas com posição do corpo válida.
# 3.	Separa os dados por pessoa (train/val/test) garantindo que cada pessoa esteja só em um conjunto.
# 4.	Extrai os frames do vídeo.
# 5.	Salva as imagens nos diretórios corretos (images/train, images/val, images/test).
# 6.	Gera os arquivos .txt de anotação no formato YOLO, normalizando para 640x640.

import pandas as pd
import os
import random
import cv2
from sklearn.model_selection import train_test_split

# === CONFIGURAÇÕES ===
VIDEO_PATH = "datasets/TownCentreXVID.mp4"  # troque para o caminho do seu vídeo
ANNOTATIONS_PATH = "datasets/annotations.txt"
CLASS_ID = 0  # pessoa
FRAME_STEP = 50

# import pdb; pdb.set_trace()

# === LEITURA DAS ANOTAÇÕES ===
df = pd.read_csv(ANNOTATIONS_PATH, header=None)
df.columns = [
    "person_id", "frame", "head_valid", "body_valid",
    "head_left", "head_top", "head_right", "head_bottom",
    "body_left", "body_top", "body_right", "body_bottom"
]

# Filtra apenas onde a posição do corpo é válida
df = df[df["body_valid"] == 1]

# Mantém apenas as colunas necessárias
df = df[["person_id", "frame", "body_left", "body_top", "body_right", "body_bottom"]]

# === SPLIT POR PESSOA (ID) ===
unique_ids = df["person_id"].unique()
random.seed(42)
random.shuffle(unique_ids)

train_ids, test_ids = train_test_split(unique_ids, test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.1765, random_state=42)  # ~15% val

def get_split(person_id):
    if person_id in train_ids:
        return "train"
    elif person_id in val_ids:
        return "val"
    else:
        return "test"

df["split"] = df["person_id"].apply(get_split)

# === CRIA DIRETÓRIOS ===
for split in ['train', 'val', 'test']:
    os.makedirs(f"dataset/images/{split}", exist_ok=True)
    os.makedirs(f"dataset/labels/{split}", exist_ok=True)

# === AGRUPA ANOTAÇÕES POR FRAME ===
grouped = df.groupby("frame")


# ABRE O VÍDEO E OBTÉM RESOLUÇÃO ORIGINAL
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolução original do vídeo: {original_width}x{original_height}")

# for frame_id in range(total_frames):
for frame_id in range(0, total_frames, FRAME_STEP):  # pula de 50 em 50 frames
    if frame_id not in grouped.groups:
        continue  # pula frames sem anotação válida

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = cap.read()
    if not success:
        print(f"Erro ao ler frame {frame_id}")
        continue


    frame_annotations = grouped.get_group(frame_id)
    split = frame_annotations.iloc[0]["split"]
    filename = f"{frame_id:06d}.jpg"
    label_path = f"datasets/labels/{split}/{filename.replace('.jpg', '.txt')}"
    image_path = f"datasets/images/{split}/{filename}"

    # Salva imagem
    cv2.imwrite(image_path, frame)

    # Salva anotação YOLO
    with open(label_path, 'w') as f:
        for _, row in frame_annotations.iterrows():
            x_center = ((row["body_left"] + row["body_right"]) / 2) / original_width
            y_center = ((row["body_top"] + row["body_bottom"]) / 2) / original_height
            width = (row["body_right"] - row["body_left"]) / original_width
            height = (row["body_bottom"] - row["body_top"]) / original_height

            if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            else:
                print(f"Anotação ignorada (fora do intervalo): frame {frame_id}")

cap.release()
print("Extração de imagens e geração de anotações finalizadas!")