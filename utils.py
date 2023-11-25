from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import os, copy, torch
from data.datasets import PACSDataset, target_domain_data, get_augmentations_linear
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from models import  LinearClassifier
import torch.nn as nn
import torch.optim as optim

def client_update(args, client_model, optimizer, train_loader, criterion,device, model_delta=None):
    client_model.train()
    total_samples = 0
    for epoch in tqdm(range(args.client_epochs)):
        for images1, images2 in train_loader:
            batch_size = images1.size(0)
            total_samples += batch_size
            images1, images2 = images1.to(device), images2.to(device)
            optimizer.zero_grad()
            z_i, z_j = client_model(images1), client_model(images2)
            loss = criterion(z_i, z_j)
            loss.backward()

            # Compute cosine similarity and update conditionally
            if model_delta is not None and args.clinet_gm == 'Delta':
                for name, param in client_model.named_parameters():
                    grad = param.grad
                    if grad is not None and name in model_delta:
                        sim = cosine_similarity(grad.view(1, -1), model_delta[name].view(1, -1))
                        if sim < 0:
                            param.grad = None  # Discard the gradient

            optimizer.step()
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

def linear_evaluation(args,global_model,device):

    # Load the saved global model
    model_path = os.path.join(args.model_save_path, f'global_model.pth')
    global_model.load_state_dict(torch.load(model_path))

    # Freeze the parameters of the model
    for param in global_model.parameters():
        param.requires_grad = False

    # Load the target domain dataset

    target_domain_dataset = target_domain_data(root=f'./data/PACS/{args.test_domain}/', transform=get_augmentations_linear())
    train_size = int(0.3 * len(target_domain_dataset))
    test_size = len(target_domain_dataset) - train_size
    train_dataset, test_dataset = random_split(target_domain_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate the linear classifier
    input_dim = 512   # Assuming the output dimension of the global model's feature extractor is 512
    num_classes = len(target_domain_dataset.classes)  # Number of classes in the target domain
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

    print(f'Accuracy of the linear classifier on the {args.test_domain} images: {100 * correct / total}%')