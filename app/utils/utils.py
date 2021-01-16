import torch
import numpy as np
import cv2
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from app.utils.mscoco_label_map import category_map


def preprocessing(img_path: str = None, img_size: int = None) -> (torch.Tensor, int, int):
    img = cv2.imread(img_path)[..., ::-1]
    img = np.array(img)/255.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_numpy = (img - mean) / std

    h, w = img_numpy.shape[:-1]
    max_size = max(h, w)

    template = np.zeros((max_size, max_size, 3))
    template[:h, :w, :] = img_numpy
    img_resized = cv2.resize(template, (img_size, img_size))
    img_tensor = torch.tensor(img_resized).float().permute(2, 0, 1).unsqueeze_(0)

    return img_tensor, h, w


def get_efficientdet(checkpoint_path):
    config = get_efficientdet_config('efficientdet_d1')
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint, strict=False)
    net = DetBenchPredict(net)
    return net


def make_predictions(model, images, score_threshold=0.45):
    predictions = []
    with torch.no_grad():
        detections = model(images)
        for i in range(images.shape[0]):
            pred = detections[i].detach().cpu().numpy()

            boxes = pred[:, :4]
            scores = pred[:, 4]
            classes = pred[:, 5]

            indexes = np.where(scores > score_threshold)[0]
            predictions.append({
                "boxes": boxes[indexes],
                "scores": scores[indexes],
                "classes": classes[indexes]
            })
    return predictions


def save_predictions(img_path, predictions, img_size):
    boxes = predictions[0]["boxes"]
    classes = predictions[0]["classes"]

    sample = cv2.imread(img_path)[..., ::-1]
    sample = sample.astype(np.float32)

    for i,  box in enumerate(boxes):
        box = box*max(sample.shape)/img_size
        box = box.astype(np.float32)
        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 4)
        cv2.putText(sample, f'{category_map[classes[i]]}', (int(box[0]),  int(box[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(sample, f'{category_map[classes[i]]}', (int(box[0]), int(box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite("app/static/pred_img.png", sample[..., ::-1])
    return
