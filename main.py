from models import SimCLR, NT_XentLoss, LinearClassifier, MoCo, SimSiam, BYOLModel
import torch
import copy
from data.datasets import PACSDataset,  get_augmentations, DomainNetDataset, HomeOfficeDataset
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset, random_split
from utils import client_update, federated_averaging, linear_evaluation, Aggregation_by_Alignment
from torchvision import datasets, models, transforms
import torch.nn as nn
import argparse
import wandb
import numpy as np
from datetime import datetime
import json

def main(args):
    
    if args.model_save_path is None:
        workdir = str(datetime.now()).replace(" ", '').replace(":", '').split(".")[0].replace("-", '')
        workdir = '_'.join((workdir , args.dataset.upper(), args.test_domain.upper() , args.backbone.lower() ,  args.client_gm , args.aggregation ,str( args.client_epochs), args.SSL))
        workdir = os.path.join('workdirs', workdir)
        os.makedirs(workdir, exist_ok=True)

        cfg_file = open(os.path.join(workdir , 'cfg.json'), 'w')
        json.dump(args.__dict__, cfg_file)
        cfg_file.close()

        args.model_save_path = workdir



    if args.wandb is not None:
        wandb.init(project='Fed_DG',name=args.wandb, config=args)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize global model with custom ResNet
    if args.backbone.lower() == 'resnet18':
        from models import ResNet18
        backbone = ResNet18()
        print('Backbone = resnet18')
    elif args.backbone.lower() == 'resnet34':
        from models import ResNet34
        backbone = ResNet34()
        print('Backbone = resnet34')
    elif args.backbone.lower() == 'resnet50':
        from models import ResNet50
        backbone = ResNet50()
        print('Backbone = resnet50')
    elif args.backbone.lower() == 'resnet101':
        from models import ResNet101
        backbone = ResNet101()
        print('Backbone = resnet101')

    if args.SSL.lower() == 'simclr':
        global_model = SimCLR(net = backbone).to(device)
        criterion = NT_XentLoss(temperature=0.5, device=device) #it uses info loss
    elif args.SSL.lower() == 'moco':
        global_model = MoCo(backbone).to(device)
        criterion = nn.CrossEntropyLoss()
    elif args.SSL.lower() == 'simsiam':
        global_model = SimSiam(backbone).to(device)
        criterion = nn.CosineSimilarity(dim=1)
    elif args.SSL.lower() == 'byol':
        global_model = BYOLModel(backbone).to(device)
        criterion = nn.CosineSimilarity(dim=1) #it uses byol loss
    else:
        print("Error: Please select one of the following SSL methods: ['SimCLR', 'MoCo', 'SimSiam', 'BYOL']")
        exit()

    # Initialize a variable to store the previous global model weights
    previous_global_model_weights = None

    assert args.dataset.lower() in ["pacs", "homeoffice", 'domainnet'], "Please make sure one of the following datasets has been selected: ['pacs', 'homeoffice', 'domainnet']"

    if args.dataset.lower() == "pacs":

        domains = ['art_painting', 'sketch', 'photo', 'cartoon']
        domains_dict = {d[0]:d for  d in domains}
        target_domain =  domains_dict[args.test_domain]
    
        assert args.test_domain.lower() in list("pacs"), "The test domain argument should be either 'P', 'A', 'C', or 'S' when dataset == pacs"
        args.test_domain = target_domain

        source_domains = [domain for domain in domains if domain not in target_domain]
        print(f"Source domains: {source_domains}",f"Target domain: {target_domain}")

        source_dataloader = {domain: DataLoader(PACSDataset(root=f'{args.dataroot}', domain=domain[0], transform=get_augmentations(dataset_name=args.dataset.lower())), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last = True) for domain in source_domains}
        total_samples = sum(len(dataset) for dataset in source_dataloader.values())
        domain_weights = {domain: len(source_dataloader[domain]) / total_samples for domain in source_domains}

    
    elif args.dataset.lower() == "domainnet":

        domains_dict = {'c': 'clipart',
                        'i': 'infograph',
                        'p': 'painting',
                        'q': 'quickdraw',
                        'r': 'real',
                        's': 'sketch'}
        target_domain =  domains_dict[args.test_domain.lower()]
    
        assert args.test_domain.lower() in list("cipqrs"), "The test domain argument should be either 'C', 'I', 'P', 'Q', 'R' or 'S' when dataset == domainnet"
        args.test_domain = target_domain

        source_domains = [domain for domain in domains_dict.keys() if domain != target_domain[0].lower()]
        print(f"Source domains: {source_domains}",f"Target domain: {target_domain}")

        source_dataloader = {domain: DataLoader(DomainNetDataset(root=f'{args.dataroot}', domain=domain[0], transform=get_augmentations(dataset_name='pacs')), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last = True) for domain in source_domains}
        total_samples = sum(len(dataset) for dataset in source_dataloader.values())
        domain_weights = {domain: len(source_dataloader[domain]) / total_samples for domain in source_domains}

    elif args.dataset.lower() == "homeoffice":
        assert args.test_domain.upper() in list("ACPR"), "The test domain argument should be either 'A', 'P', 'C', 'I' when dataset == homeoffice"
        domains_dict = {'a': 'Art', 'c': 'Clipart', 'p': 'Product', 'r': 'Real World'}

        target_domain =  domains_dict[args.test_domain.lower()]
    
        
        args.test_domain = target_domain

        source_domains = [domain for domain in domains_dict.keys() if domain != target_domain[0].lower()]
        print(f"Source domains: {source_domains}",f"Target domain: {target_domain}")

        source_dataloader = {domain: DataLoader(HomeOfficeDataset(root=f'{args.dataroot}', domain=domain[0], transform=get_augmentations(dataset_name='pacs')), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last = True) for domain in source_domains}
        total_samples = sum(len(dataset) for dataset in source_dataloader.values())
        domain_weights = {domain: len(source_dataloader[domain]) / total_samples for domain in source_domains}
 


    # Initialize client models
    client_models = {domain: copy.deepcopy(global_model) for domain in source_domains}
    if args.SSL.lower() == 'simclr':
        optimizers = {domain: optim.Adam(client_models[domain].parameters(), lr=0.0001) for domain in source_domains}
    elif args.SSL.lower() == 'moco':
        optimizers = {domain: optim.SGD(client_models[domain].parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4) for domain in source_domains}
    elif args.SSL.lower() == 'simsiam':
        optimizers = {domain: optim.SGD(client_models[domain].parameters(), lr=0.025, momentum=0.9, weight_decay=1e-4) for domain in source_domains}
    elif args.SSL.lower() == 'byol':
        optimizers = {domain: optim.SGD(client_models[domain].parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4) for domain in source_domains}
        
    # Save the initial models as epoch 0
    model_save_path = args.model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)



    print(f'Selected device: {device}')
    model_delta = None
    for comm_round in range(args.communication_rounds):
        print(f"Communication Round {comm_round + 1}/{args.communication_rounds}")

        # Train each client on its domain data
        for domain in source_domains:
            print(f"Training on domain: {domain} ...")
            client_update(args, client_models[domain], optimizers[domain], source_dataloader[domain], criterion, device, model_delta=model_delta)

        # Federated averaging with domain weights
        if args.aggregation.lower() == "fedavg":
            model_delta = federated_averaging(args, global_model, client_models, domain_weights, previous_global_model_weights)
        elif args.aggregation.lower() == "abya":
            model_delta = Aggregation_by_Alignment(args, global_model, client_models, domain_weights, previous_global_model_weights)

        if args.eval_every :
            if comm_round % args.eval_every  == 0:
                # if args.labeled_ratio_sweep:
                #     for labeled_ratio in np.linspace(0.1, 0.9, 9):
                #         linear_evaluation(args,global_model,device, labeled_ratio= labeled_ratio, comm_round = comm_round)
                # else:
                linear_evaluation(args,global_model,device, labeled_ratio= 0.1, comm_round = comm_round)
        # else:
        #     if args.wandb:
        #         wandb.log({'Accuracy': 0.0})
        
        print("############################################## End of Round ########################################")
    if args.labeled_ratio_sweep:
        for labeled_ratio in np.linspace(0.1, 0.9, 9):
            linear_evaluation(args,global_model,device, labeled_ratio= labeled_ratio)
    else:
        linear_evaluation(args,global_model,device, labeled_ratio= args.labeled_ratio)




if __name__ == '__main__':
    # Training parameters
    parser = argparse.ArgumentParser(description='Federated Domain Generalization')

    parser.add_argument('--communication_rounds', type=int, default=2, metavar='Rounds',
                        help='Number of communication rounds (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='Batch Size',
                        help='Batch size (default: 128)')
    parser.add_argument('--model_save_path', type=str, default= None, metavar='save_path',
                        help='Path to save models (default: None')
    parser.add_argument('--client_epochs', type=int, default=5, metavar='client_epochs',
                        help='Number of epochs to train on each client (default: 5)')
    parser.add_argument('--linear_lr', type=float, default=3e-3, metavar='Linear LR',
                        help='Learning rate for linear evaluation (default: 0.001)')
    parser.add_argument('--linear_epoch', type=int, default=100, metavar='Linear Epoch',
                        help='Number of epochs for linear evaluation (default: 100)')
    parser.add_argument('--aggregation', type=str, default="FedAVG", metavar='Aggregation Method',
                    help='Which aggregation method to use (default: FedAVG) [FedAVG, AbyA]')
    parser.add_argument('--client_gm', type=str, default="None", metavar='Client gradient manipulation',
                help='Which client level method to use (default: None) [None, Delta]')
    parser.add_argument('--dataset', type=str, default="pacs", metavar='Dataset to train on',
                help='Dataset to train on (default: pacs) [pacs, homeoffice]')
    parser.add_argument('--test_domain', type=str, default='p', metavar='Domain',
                        help='What domain to test on (default: photo)')
    parser.add_argument('--dataroot', type=str, default='./data/PACS/', metavar='Dataset Root Path',
                        help='The absolute path containing the selected dataset (default: photo)')
    parser.add_argument('--workers', type=int, default=2, metavar='gpu workers for the dataset',
                        help = 'help=gpu workers for the dataset (default: 2)')
    parser.add_argument('--labeled_ratio', type=float, default=0.1, metavar='labeleded ratio',
                        help = 'ratio of the labeled data in the linear evaluation')
    
    parser.add_argument('--labeled_ratio_sweep', action='store_true',
                        help = 'if selected, the labeled ratio will be sweeped from 0.1 to 1')
    
    parser.add_argument('--le_random_seed', type=int, default=42, metavar='Linear Evalutaion Random Seed',
                        help = 'The radom seed for train/test split in the linear evaluator')
    parser.add_argument('--linear_batch_size', type=int, default=512, metavar='Linear Evalutaion Batch Size',
                        help = 'Linear Evalutaion Batch Size default = 512')
    parser.add_argument('--gamma', type=float, default=0, metavar='Gamma fo AbyA',
                        help = 'Gamma fo AbyA default = 0 (0<gmma<1)')
    parser.add_argument('--abya_iter', type=int, default=3, metavar='AbyA Iteration',
                        help = 'AbyA Iteration default = 3')
    parser.add_argument('--delta_threshold', type=float, default=-0.1, metavar='Delta th',
                        help = 'Delta th default = -0.1')
    parser.add_argument('--wandb', type=str, default=None, metavar='wandb',
                        help = 'wandb run name (if None, no wandb)')
    parser.add_argument('--backbone', type=str, default='ResNet18', metavar='backbone',
                        help = 'Selects the backbone for the simclr model (default: resnet18)')
    parser.add_argument('--eval_every', type=int, default=10, metavar='eval',
                        help = 'runs the linear evaluation after every given communication rounds (default = 10), pass 0 if only want to evaluate at the end')
    parser.add_argument('--SSL', type=str, default='SimCLR', metavar='SSL',
                        help = 'Selects the SSL method (default: SimCLR) [SimCLR, MoCo]')


    args = parser.parse_args()
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    
    main(args)
