from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import os, copy, torch
from data.datasets import PACSDataset, DomainNetDataset, HomeOfficeDataset, get_augmentations_linear, get_augmentations_linear_eval
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from models import  LinearClassifier
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import info_nce_loss, loss_fn, byol_loss_fn
import wandb

def client_update(args, client_model, optimizer, train_loader, criterion,device, model_delta=None, disc_log=False):
    client_model.train()
    total_samples = 0
    for epoch in tqdm(range(args.client_epochs)):
        discard = 0
        total = 0
        for images1, images2 in train_loader:
            batch_size = images1.size(0)
            total_samples += batch_size
            images1, images2 = images1.to(device), images2.to(device)
            optimizer.zero_grad()
            
            if args.SSL.lower() == 'simclr':
                images12 = torch.cat((images1, images2), dim=0)
                features = client_model(images12)
                logits, labels = info_nce_loss(args, features, device)
                loss = loss_fn(logits, labels)

            elif args.SSL.lower() == 'moco':
                logits, labels = client_model(images1, images2)
                if logits==None:
                    continue
                loss = criterion(logits, labels)

            elif args.SSL.lower() == 'simsiam':
                p1, p2, z1, z2 = client_model(images1, images2)
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            elif args.SSL.lower() == 'byol':
                online_pred_one, target_proj_two, online_pred_two, target_proj_one = client_model(images1, images2)
                loss_one = byol_loss_fn(online_pred_one, target_proj_two)
                loss_two = byol_loss_fn(online_pred_two, target_proj_one)
                loss = loss_one + loss_two
                loss = loss.mean()
            
            loss.backward()

            # Compute cosine similarity and update conditionally
            if model_delta is not None and args.client_gm == 'Delta':
                for name, param in client_model.named_parameters():
                    grad = param.grad
                    if grad is not None and name in model_delta:
                        total += 1
                        sim = cosine_similarity(grad.view(1, -1), model_delta[name].view(1, -1))
                        if sim < args.delta_threshold:
                            param.grad = None  # Discard the gradient
                            discard += 1

            optimizer.step()
            # break
        if (model_delta is not None) and disc_log and (args.wandb is not None):
            if args.wandb:
                wandb.log({f'discard_rate': np.round(100 * discard / total, 3)})
    return client_model

def federated_averaging(args, global_model, client_models, domain_weights, previous_global_model_weights):
    global_state_dict = global_model.state_dict()
    if previous_global_model_weights is not None:
        previous_global_model_weights = copy.deepcopy(global_state_dict)

    for k in global_state_dict.keys():
        weighted_sum = sum(client_models[domain].state_dict()[k].float() * domain_weights[domain] 
                           for domain in client_models)
        global_state_dict[k] = weighted_sum
    global_model.load_state_dict(global_state_dict)
    if args.SSL.lower() == 'byol':
        for client_model in client_models.values():
            client_model.backbone.load_state_dict(global_model.backbone.state_dict())
    else:
        for client_model in client_models.values():
            client_model.load_state_dict(global_model.state_dict())
            
    torch.save(global_model.state_dict(), os.path.join(args.model_save_path, f'global_model.pth'))

    if previous_global_model_weights is not None:
        model_delta = {k: global_state_dict[k] - previous_global_model_weights[k] for k in global_state_dict}
    else:
        model_delta = None
        previous_global_model_weights = copy.deepcopy(global_state_dict)

    torch.save(previous_global_model_weights, os.path.join(args.model_save_path, f'previous_global_model.pth'))

    print("Federated aggregation completed.")
    return model_delta

def linear_evaluation(args,global_model,device, labeled_ratio = 0.1, comm_round = None):

    # Load the saved global model
    model_path = os.path.join(args.model_save_path, f'global_model.pth')
    global_model.load_state_dict(torch.load(model_path))
    global_model = global_model.to(device)

    # Freeze the parameters of the model
    for param in global_model.parameters():
        param.requires_grad = False

    # Load the target domain dataset

    if args.dataset.lower() == 'pacs':

        train_dataset = PACSDataset(root=f'{args.dataroot}', transform=get_augmentations_linear(dataset_name=args.dataset.lower()), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = True)
        test_dataset = PACSDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name=args.dataset.lower()), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = False)

        train_loader = DataLoader(train_dataset, batch_size=args.linear_batch_size, shuffle=True, num_workers=args.workers)
        test_loader = DataLoader(test_dataset, batch_size=args.linear_batch_size, shuffle=False, num_workers=args.workers)

    elif args.dataset.lower() == "domainnet":
        train_dataset = DomainNetDataset(root=f'{args.dataroot}', transform=get_augmentations_linear(dataset_name='pacs'), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = True)
        test_dataset = DomainNetDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name='pacs'), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = False)

        train_loader = DataLoader(train_dataset, batch_size=args.linear_batch_size, shuffle=True, num_workers=args.workers)
        test_loader = DataLoader(test_dataset, batch_size=args.linear_batch_size, shuffle=False, num_workers=args.workers)

    elif args.dataset.lower() == "homeoffice":
        train_dataset = HomeOfficeDataset(root=f'{args.dataroot}', transform=get_augmentations_linear(dataset_name='pacs'), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = True)
        test_dataset = HomeOfficeDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name='pacs'), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = False)

        train_loader = DataLoader(train_dataset, batch_size=args.linear_batch_size, shuffle=True, num_workers=args.workers)
        test_loader = DataLoader(test_dataset, batch_size=args.linear_batch_size, shuffle=False, num_workers=args.workers)

    

    # Instantiate the linear classifier
    input_dim = 2048   # Assuming the output dimension of the global model's feature extractor is 512
    num_classes = len(train_dataset.classes)  # Number of classes in the target domain
    linear_classifier = LinearClassifier(input_dim, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_classifier.parameters(), lr=args.linear_lr)

    # Train the linear classifier
    n_epochs = args.linear_epoch  # Define the number of epochs
    global_model.eval()  # Ensure the global model is in evaluation mode
    print("Training the linear classifier...")
    for epoch in tqdm(range(n_epochs)):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                features = global_model.backbone(images)  # Extract features from the global model

            outputs = linear_classifier(features)  # Get predictions from the linear classifier

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # break
        


    # Test the linear classifier
    linear_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = global_model.backbone(images)  # Extract features
            outputs = linear_classifier(features)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if args.wandb:
        wandb.log({f'Accuracy_lr_{labeled_ratio}': np.round(100 * correct / total, 3)}, step = comm_round)
    print(f"Labeled ratio = {labeled_ratio}")
    print(f'Accuracy of the linear classifier on the {args.test_domain} images: {np.round(100 * correct / total, 3)}%')

    logs_file = open(os.path.join(args.model_save_path, 'logs.txt'), "a")
    logs_file.write(f"Labeled ratio = {labeled_ratio} || comm round = {comm_round} || ")
    logs_file.write(f'Accuracy of the linear classifier on the {args.test_domain} images: {np.round(100 * correct / total, 3)}%')
    logs_file.write("\n")
    logs_file.close()



def Aggregation_by_Alignment(args, global_model, client_models, domain_weights, previous_global_model_weights):
    global_state_dict = global_model.state_dict()
    if previous_global_model_weights is not None:
        previous_global_model_weights = copy.deepcopy(global_state_dict)



    for k in global_state_dict.keys():
        if previous_global_model_weights is not None:
            parame_vector = torch.cat([(client_models[domain].state_dict()[k].float().flatten().unsqueeze(0)-previous_global_model_weights[k].float().flatten().unsqueeze(0))*domain_weights[domain] for domain in client_models], dim=0)
        else:
            parame_vector = torch.cat([client_models[domain].state_dict()[k].float().flatten().unsqueeze(0)*domain_weights[domain] for domain in client_models], dim=0)
        weightes = AbyA(parame_vector, num_iter=args.abya_iter, gamma=args.gamma)
        weighted_sum = sum(client_models[domain].state_dict()[k].float() * weightes[i] 
                           for i, domain in enumerate(client_models))
        global_state_dict[k] = weighted_sum
        global_state_dict[k] = weighted_sum.reshape(global_state_dict[k].shape)


    
    global_model.load_state_dict(global_state_dict)
    if args.SSL.lower() == 'byol':
        for client_model in client_models.values():
            client_model.backbone.load_state_dict(global_model.backbone.state_dict())
    else:
        for client_model in client_models.values():
            client_model.load_state_dict(global_model.state_dict())

    torch.save(global_model.state_dict(), os.path.join(args.model_save_path, f'global_model.pth'))

    if previous_global_model_weights is not None:
        model_delta = {k: global_state_dict[k] - previous_global_model_weights[k] for k in global_state_dict}
    else:
        model_delta = None
        previous_global_model_weights = copy.deepcopy(global_state_dict)

    torch.save(previous_global_model_weights, os.path.join(args.model_save_path, f'previous_global_model.pth'))

    print("Federated training completed.")
    return model_delta



def AbyA(vectors, num_iter=3, gamma=0):
    """
    Aggregates the vectors using a aggrement.

    Args:
    - vectors (torch.Tensor): The input vectors of shape (m, n).
    - num_iter (int): Number of iterations.
    - gamma (float): hyperparametr stes the weight of AbyA.

    Returns:
    - torch.Tensor: The aggregated vector (1-gmma)AbyA + gamma*avg.
    """
    # Initial aggregation as the mean of the vectors
    current_agg = torch.mean(vectors, dim=0)
    avg = current_agg.clone()
    weights = torch.ones(vectors.size(0))/vectors.size(0)
    weights = weights.to(vectors.device)

    avg_weights = weights.clone()
    for i in range(num_iter):
        # distances = torch.norm(vectors - current_agg, dim=1)
     
        cosine_similarities = torch.nn.functional.cosine_similarity(vectors, current_agg.unsqueeze(0).repeat(vectors.size(0), 1), dim=1)
        cosine_similarities = (cosine_similarities+1)/2
        # print(cosine_similarities)
        weights = torch.abs(cosine_similarities)

        # Weight inversely proportional to the distance (closer vectors have more influence)
        # weights = 1 / (distances + 1e-6)  # Adding a small value to avoid division by zero
        # dot_products = torch.matmul(vectors, current_agg)
        # weights = torch.abs(dot_products)
        # weights = torch.abs(cosine_similarities)
        
        weights /= torch.sum(weights)  # Normalize weights to sum up to 1
        
        # Update the aggregation
        current_agg = torch.sum(vectors * weights.unsqueeze(1), dim=0)
    return (1-gamma)*weights + gamma*avg_weights


