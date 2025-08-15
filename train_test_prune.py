from ultralytics import YOLO
import torch

num_epochs = 100
imgsz = 640

PRUNED_BACKBONE_WEIGHTS = "pruned_model_for_finetune.pt"

# model = YOLO("yolo11m.pt")

# 2️⃣ Substitui o backbone com os pesos podados
print('Loading pruned backbone weights...')
# backbone = model.model.model[0]
# backbone.load_state_dict(torch.load(PRUNED_BACKBONE_WEIGHTS))

# Carrega modelo PyTorch inteiro já podado
torch_model = torch.load("pruned_model_for_finetune.pt")
print('✅ Pruned backbone loaded.')


model.train(
        data="./data.yaml",
        workers=4,
        epochs=num_epochs,
        imgsz=imgsz,
        batch=16,
        device=0 ,
        project="runs/train",
        name="yolo11_oxford_tower_pruned_train"
    )

print("Training finished.")

#test
model = YOLO('/home/pesquisador/pesquisa/filipe/model_compress/runs/train/yolo11_oxford_tower_pruned_train/weights/best.pt')

# Avalia o modelo utilizando o conjunto de teste
results = model.val(data='data.yaml', split='test')

# Exibe os principais resultados
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
   


