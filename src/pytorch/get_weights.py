#!/usr/python3
import torch
import torchvision
import deeplab
from deeplab import FCN_ResNet50_Weights, DeepLabV3_ResNet50_Weights

# resnet34 for example
model = torchvision.models.resnet34(pretrained=True).to("cpu")
model.eval()
var=torch.ones((1,3,224,224))
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("resnet34.pt")

# resnet 50 for example
model = torchvision.models.resnet50(pretrained=True).to("cpu")
model.eval()
var=torch.ones((1,3,224,224))
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("resnet50.pt")

#fcn resnet50
model = deeplab.deeplabv3_resnet50(pretrained=True, weights=DeepLabV3_ResNet50_Weights.DEFAULT, progress=False).to("cpu")
model.eval()
var=torch.ones((1,3,224,224))
out = model(var)
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("deeplabv3_resnet50.pt")

#fcn resnet50
model = deeplab.fcn_resnet50(pretrained=True, weights=FCN_ResNet50_Weights.DEFAULT, progress=False).to("cpu")
model.eval()
var=torch.ones((1,3,224,224))
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("fcn_resnet50.pt")