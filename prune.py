import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

def prune_model(model, amount=0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detection_model = model.model.to(device)
    
    for name, module in detection_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
            print(f"Pruned layer: {name}")
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = YOLO('./lg_model/trained_by_yolo11x.pt').to(device)
    
    results = model.val(data='data.yaml')
    print(f'default mAP50-95: {results.box.map * 100} %')

    torch_model = model.model

    print('Pruing model...')
    pruned_torch_model = prune_model(torch_model, amount=0.15)
    print('Model pruned.')

    model.model = pruned_torch_model

    print('Saving pruned model...')
    model.save('pruned_trained_by_yolo11x.pt')
    print('Pruned model saved.')

    model = YOLO('pruned_trained_by_yolo11x.pt')
    results = model.val(data='data.yaml')
    print(f'Pruned mAP50-95 {(results.box.map * 100)} %')

if __name__ == "__main__":
    main()