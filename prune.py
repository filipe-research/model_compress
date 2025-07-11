import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO
import torch_pruning as tp

# def prune_model(model, amount=0.3):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     detection_model = model.model.to(device)
    
#     for name, module in detection_model.named_modules():
#         if isinstance(module, torch.nn.Conv2d):
#             # prune.l1_unstructured(module, name='weight', amount=amount)
#             prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)  # remove 50% dos filtros
#             prune.remove(module, 'weight')
#             # print(f"Pruned layer: {name}")
#             print(f"Structured-pruned layer: {name}")
    
#     return model

def prune_yolo_model(model, example_inputs, amount=0.5):
    model.eval()
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            # strategy = tp.strategy.LNStrategy(2)  # L2 norm pruning
            # pruning_idxs = strategy(m.weight, amount=amount)
            pruning_idxs = tp.strategy.ln_structured(m.weight, amount=amount, n=2, dim=0)
            if pruning_idxs.numel() == 0:
                continue
            plan = DG.get_pruning_plan(m, tp.prune_conv_out_channel, pruning_idxs)
            plan.exec()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #model = YOLO('./lg_model/trained_by_yolo11x.pt').to(device)
    model = YOLO('/home/pesquisador/pesquisa/filipe/model_compress/runs/train/yolo11_oxford_tower_custom_train/weights/best.pt').to(device)
    

    results = model.val(data='data.yaml', split='test')
    # print(f'default mAP50-95: {results.box.map * 100} %')
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    torch_model = model.model

    # Define dummy input for tracing
    example_input = torch.randn(1, 3, 640, 640).to(device)

    print('Pruing model...')
    # pruned_torch_model = prune_yolo_model(torch_model, amount=0.5)
    pruned_torch_model = prune_yolo_model(torch_model, example_inputs=example_input, amount=0.5)
    # prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)  # remove 50% dos filtros
    print('Model pruned.')

    model.model = pruned_torch_model

    print('Saving pruned model...')
    model.save('pruned_trained_by_yolo11x.pt')
    print('Pruned model saved.')

    model = YOLO('pruned_trained_by_yolo11x.pt')
    # results = model.val(data='data.yaml')
    results = model.val(data='data.yaml', split='test')
    # print(f'Pruned mAP50-95 {(results.box.map * 100)} %'
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

if __name__ == "__main__":
    main()