import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# select the model between the one pretrained on coco and the one pretrained on pedant
def inizializeModel(configuration):
    if configuration == 'COCO':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif configuration == 'PEDANT':
        model = getModel(2)
        model.load_state_dict(torch.load('parameters.pt'))
    else:
        return None
    model.eval()
    model.to(device)
    return model


def peopleDetection(frame, model, threshold):
    frame = F.to_tensor(frame)
    # add one size at 0 position
    frame.unsqueeze_(0)
    # model evaluation
    with torch.no_grad():
        pred = model(frame.to(device))

    peopleDetected = []

    for p in range(len(pred[0]['labels'])):
        # check prediction confidence
        if (pred[0]['scores'][p].cpu().detach().numpy() > threshold) & (
                # consider only class 'person'
                pred[0]['labels'][p].cpu().detach().numpy() == 1):
            peopleDetected.append(tuple(pred[0]['boxes'][p].cpu().detach().numpy().astype(int)))

    return peopleDetected if len(peopleDetected) != 0 else None


# create the model with the modified last layer
def getModel(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model
