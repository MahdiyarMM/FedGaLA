

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torch.nn.functional import cosine_similarity
import os
import copy
import torch.nn.functional as F


# BasicBlock, Bottleneck, and Model classes

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Model(nn.Module):
#     """ResNet18 model
#     Note two main differences from official pytorch version:
#     1. conv1 kernel size: pytorch version uses kernel_size=7
#     2. average pooling: pytorch version uses AdaptiveAvgPool
#     """

#     def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=7):
#         super(Model, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         # out = out.view(out.size(0), -1)
#         # out = self.linear(out)
#         return out


# class SimCLR(nn.Module):
#     def __init__(self, backbone):
#         super(SimCLR, self).__init__()
#         self.backbone = backbone
#         self.projector = nn.Sequential(
#             nn.Linear(512 * BasicBlock.expansion, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )

#     def forward(self, x):
#         features = self.backbone(x)
#         features = features.view(features.size(0), -1)
#         projections = self.projector(features)
#         return projections

class NT_XentLoss(nn.Module):
    def __init__(self, temperature, device):
        super(NT_XentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        # Concatenate the vectors z_i and z_j
        z = torch.cat((z_i, z_j), dim=0)

        # Cosine similarity
        sim = torch.mm(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # Assemble labels
        labels = torch.cat((torch.arange(batch_size), torch.arange(batch_size)), dim=0)
        labels = labels.to(self.device)

        # Exclude self-comparisons
        mask = ~torch.eye(2 * batch_size, dtype=bool, device=self.device)
        sim = sim.masked_select(mask).view(2 * batch_size, -1)

        loss = self.criterion(sim, labels)
        return loss



# Linear Classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.fc(x)


##############################################################################################
# simclr from easyfl


class SIMCLR_Loss(nn.Module):
    def __init__(self, temperature = 0.07, device = 'cuda:0'):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        # Concatenate the vectors z_i and z_j
        z = torch.cat((z_i, z_j), dim=0)

        # Cosine similarity
        sim = torch.mm(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # Assemble labels
        labels = torch.cat((torch.arange(batch_size), torch.arange(batch_size)), dim=0)
        labels = labels.to(self.device)

        # Exclude self-comparisons
        mask = ~torch.eye(2 * batch_size, dtype=bool, device=self.device)
        sim = sim.masked_select(mask).view(2 * batch_size, -1)

        loss = self.criterion(sim, labels)
        return loss


def info_nce_loss(args, features, device, n_views=2, temperature=0.07):
    labels = torch.cat([torch.arange(args.batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     n_views * self.conf.batch_size, n_views * self.conf.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels

loss_fn = torch.nn.CrossEntropyLoss()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet
    Note two main differences from official pytorch version:
    1. conv1 kernel size: pytorch version uses kernel_size=7
    2. average pooling: pytorch version uses AdaptiveAvgPool
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.feature_dim = 512 * block.expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d((4, 4))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


OneLayer = "1_layer"
TwoLayer = "2_layer"

RESNET18 = "resnet18"
RESNET50 = "resnet50"


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, num_layer=TwoLayer):
        super().__init__()
        self.in_features = dim
        if num_layer == OneLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == TwoLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)
    

# ------------- SimCLR Model -----------------

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()


class SimCLR(BaseModel):
    def __init__(self, net=ResNet18(), image_size=32, projection_size=2048, projection_hidden_size=4096):
        super().__init__()

        self.backbone = net
        self.backbone.fc = MLP(net.feature_dim, projection_size, projection_hidden_size)  # projector

    def forward(self, image):
        return self.backbone(image)


# ------------- MoCo Model -----------------

# class MoCo(nn.Module):
#     """
#     Build a MoCo model with: a query encoder, a key encoder, and a queue
#     https://arxiv.org/abs/1911.05722
#     """

#     def __init__(self, base_encoder=ResNet18(), dim=128, K=65536, m=0.999, T=0.07, mlp=False):
#         """
#         dim: feature dimension (default: 128)
#         K: queue size; number of negative keys (default: 65536)
#         m: moco momentum of updating key encoder (default: 0.999)
#         T: softmax temperature (default: 0.07)
#         """
#         super(MoCo, self).__init__()

#         self.K = K
#         self.m = m
#         self.T = T

#         # create the encoders
#         # num_classes is the output fc dimension

#         base_encoder.fc = nn.Linear(base_encoder.feature_dim, dim)

#         self.encoder_q = copy.deepcopy(base_encoder)
#         self.encoder_k = copy.deepcopy(base_encoder)


#         if mlp:  # hack: brute-force replacement
#             dim_mlp = self.encoder_q.fc.weight.shape[1]
#             self.encoder_q.fc = nn.Sequential(
#                 nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
#             )
#             self.encoder_k.fc = nn.Sequential(
#                 nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
#             )

#         for param_q, param_k in zip(
#             self.encoder_q.parameters(), self.encoder_k.parameters()
#         ):
#             param_k.data.copy_(param_q.data)  # initialize
#             param_k.requires_grad = False  # not update by gradient

#         # create the queue
#         self.register_buffer("queue", torch.randn(dim, K))
#         self.queue = nn.functional.normalize(self.queue, dim=0)

#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         """
#         Momentum update of the key encoder
#         """
#         for param_q, param_k in zip(
#             self.encoder_q.parameters(), self.encoder_k.parameters()
#         ):
#             param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys):
#         # gather keys before updating queue
#         keys =  concat_all_gather(keys)

#         batch_size = keys.shape[0]

#         ptr = int(self.queue_ptr)
#         assert self.K % batch_size == 0  # for simplicity

#         # replace the keys at ptr (dequeue and enqueue)
#         self.queue[:, ptr : ptr + batch_size] = keys.T
#         ptr = (ptr + batch_size) % self.K  # move pointer

#         self.queue_ptr[0] = ptr

#     @torch.no_grad()
#     def _batch_shuffle_ddp(self, x):
#         """
#         Batch shuffle, for making use of BatchNorm.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         x_gather =  concat_all_gather(x)
#         batch_size_all = x_gather.shape[0]

#         num_gpus = batch_size_all // batch_size_this

#         # random shuffle index
#         idx_shuffle = torch.randperm(batch_size_all).cuda()

#         # broadcast to all gpus
#         torch.distributed.broadcast(idx_shuffle, src=0)

#         # index for restoring
#         idx_unshuffle = torch.argsort(idx_shuffle)

#         # shuffled index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

#         return x_gather[idx_this], idx_unshuffle

#     @torch.no_grad()
#     def _batch_unshuffle_ddp(self, x, idx_unshuffle):
#         """
#         Undo batch shuffle.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         x_gather = concat_all_gather(x)
#         batch_size_all = x_gather.shape[0]

#         num_gpus = batch_size_all // batch_size_this

#         # restored index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

#         return x_gather[idx_this]

#     def forward(self, im_q, im_k):
#         """
#         Input:
#             im_q: a batch of query images
#             im_k: a batch of key images
#         Output:
#             logits, targets
#         """

#         # compute query features
#         q = self.encoder_q(im_q)  # queries: NxC
#         q = nn.functional.normalize(q, dim=1)

#         # compute key features
#         with torch.no_grad():  # no gradient to keys
#             self._momentum_update_key_encoder()  # update the key encoder

#             # shuffle for making use of BN
#             im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

#             k = self.encoder_k(im_k)  # keys: NxC
#             k = nn.functional.normalize(k, dim=1)

#             # undo shuffle
#             k = self._batch_unshuffle_ddp(k, idx_unshuffle)

#         # compute logits
#         # Einstein sum is more intuitive
#         # positive logits: Nx1
#         l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
#         # negative logits: NxK
#         l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

#         # logits: Nx(1+K)
#         logits = torch.cat([l_pos, l_neg], dim=1)

#         # apply temperature
#         logits /= self.T

#         # labels: positive key indicators
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

#         # dequeue and enqueue
#         self._dequeue_and_enqueue(k)

#         return logits, labels
    
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder=ResNet18(), dim=2048, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        encoder_q is replaced with backbone for consistency with SimCLR
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        base_encoder.fc = nn.Linear(base_encoder.feature_dim, dim)

        self.backbone = copy.deepcopy(base_encoder)
        self.encoder_k = copy.deepcopy(base_encoder)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.backbone.fc.weight.shape[1]
            self.backbone.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.backbone.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.backbone.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.backbone(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        try:
            self._dequeue_and_enqueue(k)
        except:
            return None, None

        return logits, labels
#Moco functions
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output