from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder


def get_augmentations(dataset_name = 'pacs'):
    if dataset_name == "pacs":
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        return augmentation

def get_augmentations_linear(dataset_name = 'pacs'):
    if dataset_name == "pacs":
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
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
