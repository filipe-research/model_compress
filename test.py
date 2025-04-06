from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO('/home/pesquisador/pesquisa/filipe/model_compress/runs/train/yolo11_oxford_tower_custom_train7/weights/best.pt')

# Avalia o modelo utilizando o conjunto de teste
results = model.val(data='data.yaml', split='test')

# Exibe os principais resultados
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")