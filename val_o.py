#%% 
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import timm
import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader


class ImageNetODataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

transform = transforms.Compose([
    transforms.Resize(256),              # resize the input image to 256x256
    transforms.CenterCrop(224),          # crop the center 224x224 region of the image
    transforms.ToTensor(),               # convert the image to a PyTorch tensor
    transforms.Normalize(                # normalize the image
        mean=[0.485, 0.456, 0.406],       # using the ImageNet mean and standard deviation
        std=[0.229, 0.224, 0.225]
    )
])

# dataset = ImageNetODataset('~/scratch.cmsc663/imagenet-o/', transform=transform)

val_set = ImageNet(
    root='~/scratch.cmsc663/',          # path to the ImageNet validation set
    split='val',                          # use the validation split of the dataset
    transform=transform                   # apply the specified transform to each image
)

dataloader = DataLoader(val_set, batch_size=64, shuffle=True)

model = timm.create_model("hf_hub:timm/vit_small_patch32_224.augreg_in21k_ft_in1k", pretrained=True).cuda()
model.eval()
with torch.no_grad():
    total_correct = 0
    total_images = 0
    for images, labels in dataloader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_images += labels.size(0)
        total_correct += (predicted == labels).sum().item()
    accuracy = 100 * total_correct / total_images
    print('Accuracy on ImageNet dataset: {:.2f}%'.format(accuracy))
#%% 
