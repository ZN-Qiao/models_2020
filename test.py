from torch import nn
import torchvision.models as models
import copy

class celu_smooth_relu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_celu = F.celu(x)

        x_gap = F.relu(x) - x_celu

        x_out = x_celu + x_gap.data

        return x_out

model = models.resnet50(pretrained=False)

model_celu = copy.deepcopy(model)
model_celu.layer3[1].relu = celu_smooth_relu()
model_celu.layer3[2].relu = celu_smooth_relu()
model_celu.layer3[3].relu = celu_smooth_relu()
model_celu.layer3[4].relu = celu_smooth_relu()
model_celu.layer3[5].relu = celu_smooth_relu()
model_celu.layer4[0].relu = celu_smooth_relu()
model_celu.layer4[1].relu = celu_smooth_relu()
model_celu.layer4[2].relu = celu_smooth_relu()

# for name, layer in model_celu.named_modules():
#     # if isinstance(layer, nn.ReLU):
#         print(name, layer)