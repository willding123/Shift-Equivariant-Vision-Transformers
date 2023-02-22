import torchvision
train = torchvision.datasets.ImageNet("~/scratch.cmsc663", "train")
test = torchvision.datasets.ImageNet("~/scratch.cmsc663", "val")
