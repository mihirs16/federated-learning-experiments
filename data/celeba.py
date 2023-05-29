import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Root directory for dataset
dataroot = "/import/network-temp/ahmed/fedmtda/celeba/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Initialize the dataset
dataset = dset.ImageFolder(
	root=dataroot,
	transform=(transforms.Compose([
		transforms.Resize(image_size)
		transform.CenterCrop(image_size)
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])	
)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(
	dataset, 
	batch_size=batch_size,
	shuffle=True,
	num_workers=num_workers
)

