# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
from torch import nn
import torchvision.transforms.functional as F
from prefetch_generator import BackgroundGenerator
import tqdm
import copy
from copy import deepcopy

from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum


class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(256, 512, 256)
        self.prediction_head = BYOLPredictionHead(256, 512, 256)

        self.backbone_momentum = deepcopy(self.backbone)
        self.projection_head_momentum = deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
                                                   # x: 32, 3, 640, 640
        y = self.backbone(x).flatten(start_dim=1)  # output y: 32, 1
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


def normalize_tensor(tensor, mean, std, inplace=False):
    return F.normalize(tensor, mean, std, inplace)

if __name__ == "__main__":
    import torchvision.transforms as transforms
    import argparse
    from models.common import *
    from models.experimental import *
    from utils.autoanchor import check_anchor_order
    from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
    from utils.plots import feature_visualization
    from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                                time_sync)
    from models.yolo import YoloBackbone

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./models/segment/yolov5l-seg.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))

    # Create model
    yoloBackbone = YoloBackbone(opt.cfg)

    model = BYOL(yoloBackbone)
    gpus = [0, 1, 2, 3]
    # torch.cuda.set_device("cuda:{}".format(gpus[0]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    model.to(device)
    # cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
    # dataset = LightlyDataset.from_torch_dataset(cifar10)
    # or create a dataset from a folder containing images or videos:
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    # dataset = LightlyDataset("/data/ssl")
    dataset = LightlyDataset("./datasets/OrCaScoreDataSet/images/train2017")

    collate_fn = SimCLRCollateFunction(
    input_size=640,
    gaussian_blur=0.,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    criterion = NegativeCosineSimilarity()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    print("Starting Training")
    best_avg_loss = 1000000
    for epoch in range(300):
        total_loss = 0
        for (x0, x1), _, _ in tqdm.tqdm(BackgroundGenerator(dataloader)):
            update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
            x0 = x0.to(device)
            x1 = x1.to(device)
            p0 = model(x0)
            z0 = model.forward_momentum(x0)
            p1 = model(x1)
            z1 = model.forward_momentum(x1)
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

        if best_avg_loss > avg_loss:
            torch.save(model.backbone.state_dict(), "best_seg_yolopBackbone.pth")
            print(f"Finding optimal model params. Loss is dropping from {best_avg_loss:.4f} to {avg_loss:.4f}")
            best_avg_loss = avg_loss
