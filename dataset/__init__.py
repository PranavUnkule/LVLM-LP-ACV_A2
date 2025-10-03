dataset_roots = {
    "VizWiz": "/workspace/LVLM-LP/data/VizWiz/",
    "MMSafety": "/workspace/LVLM-LP/data/qinyu/data/MM-SafetyBench/",
    "MAD": "/workspace/LVLM-LP/data/coco/",   # MADBench uses COCO images
    "MathVista": "/workspace/LVLM-LP/data/MathVista/",
    "POPE": "/workspace/LVLM-LP/data/coco/",  # POPE uses COCO images
    "ImageNet": "/workspace/LVLM-LP/data/ImageNet/"
}


def build_dataset(dataset_name, split, prompter):
    if dataset_name == "VizWiz":
        from .VizWiz import VizWizDataset
        dataset = VizWizDataset(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "MAD":
        from .MADBench import MADBench
        dataset = MADBench(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "ImageNet":
        from .ImageNet import ImageNetDataset
        dataset = ImageNetDataset(split, dataset_roots[dataset_name])
    elif dataset_name == "MathVista":
        from .MathV import MathVista
        dataset = MathVista(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "MMSafety":
        from .MMSafety import MMSafetyBench
        dataset = MMSafetyBench(prompter, split, dataset_roots[dataset_name])
    elif dataset_name == "POPE":
        from .POPE import POPEDataset
        dataset = POPEDataset(split, dataset_roots[dataset_name])
    else:
        from .base import BaseDataset
        dataset = BaseDataset()
        
    return dataset.get_data()
