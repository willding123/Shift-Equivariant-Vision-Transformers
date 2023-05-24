import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Dataloader Timing')
    parser.add_argument('--num_cpus', default=4, type=int,
                        metavar='N', help='number of cpus (default: 4)')
    args = parser.parse_args()
    return args

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "~/scratch.cmsc663/train"
    batch_size = 256
    pin_memory = True  # set to True if using a GPU

    train_dataset = ImageFolder(data_dir, transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    n1 = args.num_cpus
    n2 = args.num_cpus * 2
    n4 = args.num_cpus * 4
    n8 = args.num_cpus * 8
    for num_workers in [n1, n2, n4, n8]:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )

        total_data_loading_time = 0
        total_data_to_gpu_time = 0
        num_batches = 5  
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == num_batches:
                break
            
            # mv_start_time = time.time()
            # # move data to gpu, non-blocking
            data = data.to(device, non_blocking=True)
            # mv_end_time = time.time()
            
            # total_data_to_gpu_time += mv_end_time - mv_start_time
        total_data_loading_time = time.time() - start_time
            
        print(f"Number of workers: {num_workers}, "
            f"Average Data loading time: {total_data_loading_time / num_batches :.5f} seconds, ")
            # f"Average data to GPU time per batch: {total_data_to_gpu_time / num_batches :.5f} seconds")
        # clear cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args)