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
    
def get_augmentations_linear_eval(dataset_name = 'pacs'):
    if dataset_name == "pacs":
    
        augmentation = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
        return augmentation

# class PACSDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.dataset = ImageFolder(root=root)
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         image, _ = self.dataset[idx]
#         image1 = self.transform(image) if self.transform else image
#         image2 = self.transform(image) if self.transform else image
#         return image1, image2

# def load_pacs_dataset(domain):
#     dataset_path = f'./PACS/{domain}/'
#     dataset = PACSDataset(root=dataset_path, transform=get_augmentations())
#     return DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    
# def target_domain_data(root,transform):
#     return ImageFolder(root=root, transform = transform)



import numpy as np
import cv2
from glob import glob
import pandas as pd
import os
from PIL import Image
from data.datasets import get_augmentations, get_augmentations_linear_eval
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from copy import deepcopy

def show_tensor_img(img):
    img = img.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    img = img.permute(1,2,0)

    img = img - img.min()
    img = img / img.max()

    return img

class PACSDataset(Dataset):

    """PACS Dataset"""

    def __init__(self, root = "/media/milad/DATA/Federated/data/PACS", transform=None, domain = "A", labeled_ratio = None, linear_train = True):
        """
        Arguments:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.linear_train = linear_train
        self.transform = transform
        self.domains_dict = {"A" : "art_painting", "C": "cartoon", "P": "photo", "S" :"sketch"}
        self.df = pd.read_csv(os.path.join(root, 'pacs.csv'))
        self.df = self.df[self.df['domain'] == self.domains_dict[domain.upper()]]
        
    
        self.classes = os.listdir(os.path.join(root , self.domains_dict[domain.upper()]))
        self.numerical2cat_dict = {self.classes.index(cat) : cat for cat in self.classes}
        self.cat2numerical_dict = {v:k for k,v in self.numerical2cat_dict.items()}


        self.label_ratio = labeled_ratio
        if self.label_ratio is not None:
            if self.linear_train:
                self.df = self.df.iloc[:int(self.label_ratio * len(self.df))]
            else:
                self.df = self.df.iloc[int(self.label_ratio * len(self.df)):]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        

        img, domain, cat = self.df['data'].iloc[idx], self.df['domain'].iloc[idx], self.df['cat'].iloc[idx]

        if self.label_ratio is None:
            img1 = Image.open(os.path.join(self.root, img))
            img2 = deepcopy(img1)

            # img1.show()
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return img1, img2
        
        else:
            img1 = Image.open(os.path.join(self.root, img))
            if self.transform is not None:
                img1 = self.transform(img1)
            return img1, self.cat2numerical_dict[cat]
        




def create_dataset_df(root_dir, seed = 42, labeled_ratio = 0.1, test_domain = "real"):
    domains_dict = {k[0]:k for k in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, k))}
    categories_list = os.listdir(os.path.join(root_dir , list(domains_dict.values())[0]))
    imgs_names = []
    domains = []
    categories = []
    for domain in domains_dict.values():
        for cat in categories_list:
            path = os.path.join(root_dir, domain, cat)
            files = glob(os.path.join(path, "*"))
            imgs_names.extend(files)
            domains.extend(len(files) * [domain])
            categories.extend(len(files) * [cat])

    df = pd.DataFrame(data = {"data" : imgs_names, "domain": domains, "cat" :  categories})
    df['data'] = df['data'].apply(lambda x: x.replace(root_dir, ""))
    sampled_df = df[df['domain'] == test_domain].groupby('cat', group_keys=False).apply(lambda x: x.sample(frac=labeled_ratio, random_state=seed))

    if labeled_ratio:
        assert len(sampled_df)


    train_df = df.drop(sampled_df.index)

    return train_df, sampled_df




# class DomainNetDataset(Dataset):

#     """DomainNet Dataset"""

#     def __init__(self, root = "/media/milad/DATA/Federated/data/DomainNet", transform=None, domain = "R", labeled_ratio = None, linear_train = True, random_seed = 42):
#         """
#         Arguments:
#             root (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         if labeled_ratio is None:
#             linear_train = False
#         self.root = root
#         self.linear_train = linear_train
#         self.transform = transform
#         self.domains_dict = {k[0]:k for k in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, k))}
#         self.categories_list = os.listdir(os.path.join(root , list(self.domains_dict.values())[0]))
        
    
#         self.classes = os.listdir(os.path.join(root , self.domains_dict[domain[0].lower()]))
#         self.numerical2cat_dict = {self.classes.index(cat) : cat for cat in self.classes}
#         self.cat2numerical_dict = {v:k for k,v in self.numerical2cat_dict.items()}

#         self.train_df , self.test_df = create_dataset_df(root_dir=root, seed = random_seed, labeled_ratio = (0 if labeled_ratio is None else labeled_ratio), test_domain = self.domains_dict[domain.lower()[0]])


#         assert(len(self.train_df))

#         self.label_ratio = labeled_ratio
#         # if self.label_ratio is not None:
#         if self.linear_train:
#             self.df = self.test_df
#         else:
#             self.df = self.train_df
        
#         self.df = self.df[self.df['domain'] == self.domains_dict[domain.lower()[0]]]

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):

        

#         img, domain, cat = self.df['data'].iloc[idx], self.df['domain'].iloc[idx], self.df['cat'].iloc[idx]

#         if self.label_ratio is None:
#             img1 = Image.open(os.path.join(self.root, img))
#             img2 = deepcopy(img1)

#             if self.transform is not None:
#                 img1 = self.transform(img1)
#                 img2 = self.transform(img2)
#             return img1, img2
        
#         else:
#             img1 = Image.open(os.path.join(self.root, img))
#             if self.transform is not None:
#                 img1 = self.transform(img1)
#             return img1, self.cat2numerical_dict[cat]
        
class DomainNetDataset(Dataset):

    """DomainNet Dataset"""

    def __init__(self, root = "/media/milad/DATA/Federated/data/DomainNet", transform=None, domain = "R", labeled_ratio = None, linear_train = True, random_seed = 42):
        """
        Arguments:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if labeled_ratio is None:
            linear_train = False
        self.root = root
        self.linear_train = linear_train
        self.transform = transform
        self.domains_dict = {k[0]:k for k in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, k))}
        self.categories_list = os.listdir(os.path.join(root , list(self.domains_dict.values())[0]))
        
    
        self.classes = os.listdir(os.path.join(root , self.domains_dict[domain[0].lower()]))
        self.numerical2cat_dict = {self.classes.index(cat) : cat for cat in self.classes}
        self.cat2numerical_dict = {v:k for k,v in self.numerical2cat_dict.items()}

        self.train_df , self.test_df = create_dataset_df(root_dir=root, seed = random_seed, labeled_ratio = (0 if labeled_ratio is None else labeled_ratio), test_domain = self.domains_dict[domain.lower()[0]])


        assert(len(self.train_df))

        self.label_ratio = labeled_ratio
        # if self.label_ratio is not None:
        if self.linear_train:
            self.df = self.test_df
        else:
            self.df = self.train_df
        
        self.df = self.df[self.df['domain'] == self.domains_dict[domain.lower()[0]]]
        self.img_list = []
        print(f"Loading images for domain {domain}")
        for idx, img_name in enumerate(self.df['data']):
            img = Image.open(os.path.join(self.root, img_name))
            self.img_list.append(img)
            if idx % 30000 == 0:
                print(f'{idx} images have been loaded')
        print('finished loading')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img, domain, cat = self.df['data'].iloc[idx], self.df['domain'].iloc[idx], self.df['cat'].iloc[idx]

        if self.label_ratio is None:
            # img1 = Image.open(os.path.join(self.root, img))
            img1 = self.img_list[idx]
            img2 = deepcopy(img1)

            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return img1, img2
        
        else:
            img1 = Image.open(os.path.join(self.root, img))
            if self.transform is not None:
                img1 = self.transform(img1)
            return img1, self.cat2numerical_dict[cat]


class HomeOfficeDataset(Dataset):

    """Home Office Dataset"""

    def __init__(self, root = "/media/milad/DATA/Federated/data/OfficeHomeDataset/", transform=None, domain = "R", labeled_ratio = None, linear_train = True, random_seed = 42):
        """
        Arguments:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if labeled_ratio is None:
            linear_train = False
        self.root = root
        self.linear_train = linear_train
        self.transform = transform
        self.domains_dict = {k[0].lower():k for k in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, k))}
        self.categories_list = os.listdir(os.path.join(root , list(self.domains_dict.values())[0]))
        
    
        self.classes = os.listdir(os.path.join(root , self.domains_dict[domain[0].lower()]))
        self.numerical2cat_dict = {self.classes.index(cat) : cat for cat in self.classes}
        self.cat2numerical_dict = {v:k for k,v in self.numerical2cat_dict.items()}



        self.train_df , self.test_df = create_dataset_df(root_dir=root, seed = random_seed, labeled_ratio = (0 if labeled_ratio is None else labeled_ratio), test_domain = self.domains_dict[domain.lower()[0]])


        assert(len(self.train_df))

        self.label_ratio = labeled_ratio
        # if self.label_ratio is not None:
        if self.linear_train:
            self.df = self.test_df
        else:
            self.df = self.train_df
        
        self.df = self.df[self.df['domain'] == self.domains_dict[domain.lower()[0]]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        

        img, domain, cat = self.df['data'].iloc[idx], self.df['domain'].iloc[idx], self.df['cat'].iloc[idx]
        # print(os.path.join(self.root, img))
        if self.label_ratio is None:
            img1 = Image.open(os.path.join(self.root, img))
            img2 = deepcopy(img1)

            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return img1, img2
        
        else:
            img1 = Image.open(os.path.join(self.root, img))
            if self.transform is not None:
                img1 = self.transform(img1)
            return img1, self.cat2numerical_dict[cat]