from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import os, copy, torch
from data.datasets import PACSDataset, DomainNetDataset, HomeOfficeDataset, TerraIncDataset, get_augmentations_linear, get_augmentations_linear_eval
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from models import  LinearClassifier
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import info_nce_loss, loss_fn, byol_loss_fn
import wandb


def client_update(args, client_model, optimizer, train_loader, criterion,device, model_delta=None, disc_log=False, update_delta = False):
    if args.prox_mu:
        initial_model = copy.deepcopy(client_model) # the model at the begining of the communication round (for fedprox)
        
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
            
            # added fedprox proximity term
            if args.prox_mu:
                proximity_term = 0
                for w0, w1 in zip(client_model.parameters(), initial_model.parameters()):
                    proximity_term += (w0 - w1).norm(2) * args.prox_mu 
                # print(f'loss: {loss} ---- prox_term : {proximity_term}')
                loss += proximity_term
            loss.backward()

            # Compute cosine similarity and update conditionally
            if model_delta is not None and args.client_gm.lower() == 'la':
                for name, param in client_model.named_parameters():
                    grad = param.grad
                    if grad is not None and name in model_delta:
                        total += 1
                        sim = cosine_similarity(grad.view(1, -1), model_delta[name].view(1, -1))
                        print(sim)
                        if sim < args.local_threshold:
                            print(param.grad.shape)
                            print(param.grad.dtype)
                            param.grad = None  # Discard the gradient
                            discard += 1
            

            optimizer.step()
            # break
        if (model_delta is not None) and disc_log and (args.wandb is not None):
            if args.wandb:
                wandb.log({f'discard_rate': np.round(100 * discard / total, 3)})
        # print('discard : ', discard)
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
        train_dataset = DomainNetDataset(root=f'{args.dataroot}', transform=get_augmentations_linear(dataset_name=args.dataset.lower()), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = True)
        test_dataset = DomainNetDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name=args.dataset.lower()), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = False)

        train_loader = DataLoader(train_dataset, batch_size=args.linear_batch_size, shuffle=True, num_workers=args.workers)
        test_loader = DataLoader(test_dataset, batch_size=args.linear_batch_size, shuffle=False, num_workers=args.workers)

    elif args.dataset.lower() == "minidomainnet":
        train_dataset = DomainNetDataset(root=f'{args.dataroot}', transform=get_augmentations_linear(dataset_name=args.dataset.lower()), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = True)
        test_dataset = DomainNetDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name=args.dataset.lower()), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = False)

        train_loader = DataLoader(train_dataset, batch_size=args.linear_batch_size, shuffle=True, num_workers=args.workers)
        test_loader = DataLoader(test_dataset, batch_size=args.linear_batch_size, shuffle=False, num_workers=args.workers)

    elif args.dataset.lower() == "homeoffice":
        train_dataset = HomeOfficeDataset(root=f'{args.dataroot}', transform=get_augmentations_linear(dataset_name='pacs'), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = True)
        test_dataset = HomeOfficeDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name='pacs'), domain=args.test_domain[0], labeled_ratio= labeled_ratio, linear_train = False)

        train_loader = DataLoader(train_dataset, batch_size=args.linear_batch_size, shuffle=True, num_workers=args.workers)
        test_loader = DataLoader(test_dataset, batch_size=args.linear_batch_size, shuffle=False, num_workers=args.workers)
        
    elif args.dataset.lower() == "terrainc":
        train_dataset = TerraIncDataset(root=f'{args.dataroot}', transform=get_augmentations_linear(dataset_name='pacs'), domain=args.test_domain.split("_")[-1], labeled_ratio= labeled_ratio, linear_train = True)
        test_dataset = TerraIncDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name='pacs'), domain=args.test_domain.split("_")[-1], labeled_ratio= labeled_ratio, linear_train = False)

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
        weightes = Global_Alignment(parame_vector, num_iter=args.ga_iter, gamma=args.gamma)
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



def Global_Alignment(vectors, num_iter=3, gamma=0):
    """
    Aggregates the vectors using a aggrement.

    Args:
    - vectors (torch.Tensor): The input vectors of shape (m, n).
    - num_iter (int): Number of iterations.
    - gamma (float): hyperparametr stes the weight of AbyA.

    Returns:
    - torch.Tensor: The aggregated vector (1-gmma)*GA + gamma*avg.
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


def FedEMA(args, global_model, client_models, domain_weights, previous_global_model_weights):
    if args.SSL.lower() != 'byol':
        raise ValueError('FedEMA is only supported for BYOL')
    
    if args.FedEMA_abya:
        global_state_dict = global_model.state_dict()
        if previous_global_model_weights is not None:
            previous_global_model_weights = copy.deepcopy(global_state_dict)



        for k in global_state_dict.keys():
            if previous_global_model_weights is not None:
                parame_vector = torch.cat([(client_models[domain].state_dict()[k].float().flatten().unsqueeze(0)-previous_global_model_weights[k].float().flatten().unsqueeze(0))*domain_weights[domain] for domain in client_models], dim=0)
            else:
                parame_vector = torch.cat([client_models[domain].state_dict()[k].float().flatten().unsqueeze(0)*domain_weights[domain] for domain in client_models], dim=0)
            weightes = Global_Alignment(parame_vector, num_iter=args.abya_iter, gamma=args.gamma)
            weighted_sum = sum(client_models[domain].state_dict()[k].float() * weightes[i] 
                            for i, domain in enumerate(client_models))
            global_state_dict[k] = weighted_sum
            global_state_dict[k] = weighted_sum.reshape(global_state_dict[k].shape)


        
        global_model.load_state_dict(global_state_dict)


    else:
        global_state_dict = global_model.state_dict()
        if previous_global_model_weights is not None:
            previous_global_model_weights = copy.deepcopy(global_state_dict)

        # Calculating FedAVG
        for k in global_state_dict.keys():
            weighted_sum = sum(client_models[domain].state_dict()[k].float() * domain_weights[domain] 
                            for domain in client_models)
            global_state_dict[k] = weighted_sum
        global_model.load_state_dict(global_state_dict)
    
    # Calculating Lambda
    lambda_ = {}
    for domain, client_model in client_models.items():
        flattened_params_global = torch.nn.utils.parameters_to_vector(global_model.backbone.parameters())
        flattened_params_local = torch.nn.utils.parameters_to_vector(client_model.backbone.parameters())

        lambda_[domain] = args.FedEMA_tau/(torch.norm(flattened_params_global-flattened_params_local)+1e-6)

    # Calculating Mu
    mu = {}
    for k,v in lambda_.items():
        if v>1:
            mu[k] = 1
        else:
            mu[k] = v
    

    
    for domain, client_model in client_models.items():
        new_state_dict = copy.deepcopy(client_model.state_dict())

        for k in global_state_dict.keys():
            new_state_dict[k] = (mu[domain])*new_state_dict[k] + (1-mu[domain])*global_state_dict[k]
        
        client_model.load_state_dict(new_state_dict)



            
    torch.save(global_model.state_dict(), os.path.join(args.model_save_path, f'global_model.pth'))

    if previous_global_model_weights is not None:
        model_delta = {k: global_state_dict[k] - previous_global_model_weights[k] for k in global_state_dict}
    else:
        model_delta = None
        previous_global_model_weights = copy.deepcopy(global_state_dict)

    torch.save(previous_global_model_weights, os.path.join(args.model_save_path, f'previous_global_model.pth'))

    print("Federated aggregation completed.")
    return model_delta





from torch.utils.data import DataLoader, TensorDataset
def linear_evaluation(args, global_model, device, labeled_ratio=0.1, comm_round=None, linear_weights=None, lr=0.0001, verbose=False):
    global_model = copy.deepcopy(global_model)
    # Freeze the parameters of the model
    for param in global_model.parameters():
        param.requires_grad = False

    # Load the target domain dataset
    if args.dataset.lower() == 'pacs':
        train_dataset = PACSDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name=args.dataset.lower()), domain=args.test_domain[0], labeled_ratio=labeled_ratio, linear_train=True)
        test_dataset = PACSDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name=args.dataset.lower()), domain=args.test_domain[0], labeled_ratio=labeled_ratio, linear_train=False)

    elif args.dataset.lower() == "terrainc":
        # print('Loading Terrainc')
        train_dataset = TerraIncDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name='pacs'), domain=args.test_domain.split("_")[-1], labeled_ratio= labeled_ratio, linear_train = True)
        test_dataset = TerraIncDataset(root=f'{args.dataroot}', transform=get_augmentations_linear_eval(dataset_name='pacs'), domain=args.test_domain.split("_")[-1], labeled_ratio= labeled_ratio, linear_train = False)

        train_loader = DataLoader(train_dataset, batch_size=args.linear_batch_size, shuffle=True, num_workers=args.workers)
        test_loader = DataLoader(test_dataset, batch_size=args.linear_batch_size, shuffle=False, num_workers=args.workers)

    # Preprocess datasets to extract features once
    train_loader = DataLoader(train_dataset, batch_size=args.linear_batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.linear_batch_size, shuffle=False, num_workers=args.workers)

    train_features, train_labels = extract_features(global_model, train_loader, device)
    test_features, test_labels = extract_features(global_model, test_loader, device)

    # Convert to TensorDatasets and DataLoaders for training/testing the linear classifier
    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=args.linear_batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=args.linear_batch_size, shuffle=False)

    # Instantiate the linear classifier
    input_dim = 2048  # Assuming the output dimension of the global model's feature extractor is 2048
    num_classes = len(train_dataset.classes)  # Number of classes in the target domain
    
    linear_classifier = LinearClassifier(input_dim, num_classes).to(device)
    if linear_weights is not None:
        linear_classifier.load_state_dict(copy.deepcopy(linear_weights))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.3, last_epoch=-1, verbose=verbose)

    # Train the linear classifier
    # print("Training the linear classifier...")
    accs_list = []
    for epoch in tqdm(range(100), disable=not verbose):
        linear_classifier.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = linear_classifier(features)  # Get predictions from the linear classifier

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        acc = evaluate(linear_classifier, test_loader, device)
        if verbose:
            print(f'Epoch: {epoch}, Test Accuracy: {acc}%')
        accs_list.append(acc)
    
    return  accs_list

def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(device)
            output = model.backbone(images)
            features.append(output)
            labels.append(labels_batch)

    return torch.cat(features), torch.cat(labels)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def linear_eval_repeat(args, global_model, device, labeled_ratio=0.1, comm_round=None, linear_weights=None, lr=0.0001, verbose=False, repeat = 10):
    acc_list = []
    for i in range(repeat):
        acc = linear_evaluation(args, global_model, device, labeled_ratio=labeled_ratio, comm_round=comm_round, 
                                linear_weights=linear_weights, lr=lr, verbose=verbose)[-1]
        
        acc_list.append(acc)
    print(f'Acc : {np.mean(acc_list)}')
    return acc_list
