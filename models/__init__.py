from .clip_models import CLIPModel
from .ablation import ablation1, ablation2, ablation3
from .imagenet_models import ImagenetModel


VALID_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',

    'CLIP:RN50', 
    'CLIP:RN101', 
    'CLIP:RN50x4', 
    'CLIP:RN50x16', 
    'CLIP:RN50x64', 
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
    'CLIP:ViT-L/14@336px',
]

def get_model(name, num_classes, select_k, training, p, ablation_opt):
    assert name in VALID_NAMES
    if ablation_opt == 1:
        return ablation1.CLIPModel(name[5:], num_classes, select_k, training, p)
    elif ablation_opt == 2:
        return ablation2.CLIPModel(name[5:], num_classes, select_k, training, p)
    elif ablation_opt == 3:
        return ablation3.CLIPModel(name[5:], num_classes, select_k, training, p)
    else:
        if name.startswith("Imagenet:"):
            # return ImagenetModel(name[9:])
            return CLIPModel(name[5:], num_classes, select_k, training, p)
        elif name.startswith("CLIP:"):
            return CLIPModel(name[5:], num_classes, select_k, training, p)
        else:
            assert False
