from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder


def get_augmentations(dataset_name = 'pacs'):
    if dataset_name == "pacs":
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize((32,32),antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return augmentation

def get_augmentations_linear(dataset_name = 'pacs'):
    if dataset_name == "pacs":
        augmentation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32),antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return augmentation

class PACSDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        image1 = self.transform(image) if self.transform else image
        image2 = self.transform(image) if self.transform else image
        return image1, image2

def load_pacs_dataset(domain):
    dataset_path = f'./PACS/{domain}/'
    dataset = PACSDataset(root=dataset_path, transform=get_augmentations())
    return DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    
def target_domain_data(root,transform):
    return ImageFolder(root=root, transform = transform)
