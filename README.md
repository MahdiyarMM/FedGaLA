# FedGaLa
Federated Unsupervised Domain Generalization using Global and Local
Alignment of Gradients

## Introduction
This is the officiial documentation for our FedGaLa paper submitted to ICML 2024. 
Our proposed FedGaLA performs gradient alignment at both local (client) and global (server) stages to achieve a more generalized aggregated model.

![fig_1_pages-to-jpg-0001](https://github.com/MahdiyarMM/Fed_DG/assets/44018277/2ca2a417-e3f6-4566-8e0e-c9af6e112c80)


## Requirements
To train one of the baselines, please follow the instructions below

1) Install the requirements through the code below:
   `pip install -r requirements.txt`
2) Download the datasets from the corresponding links in section data and extract to ./data folder
3) Select the SSL method you would like to use from `byol`, `simsiam`, `moco` or `simclr`.
4) run the following code to train the baseline model with the selected SSL method. The example given below is for fedsimclr model on the PACS dataset.



```bash
python main.py  --test_domain s --dataset pacs --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 7 --backbone resnet18 --aggregation FedAVG --SSL simclr --labeled_ratio 0.3 --workers 2
```

 
To train FedGaLa, run the following code:
```bash
python main.py  --test_domain s --dataset pacs --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 7 --backbone resnet18 --aggregation GA --SSL simclr --labeled_ratio 0.3  --client_gm LA --local_threshold 0.0 --gamma 0.00 --workers 2
```
In the code above, `GA` and `LA` stand for Global Alignment and Local Alignment, respectively. By default, aggregation is set to FedAVG and client_gm is None. For workers, plase set it to `0` if you are using a windows machine.

## Data
We train on the following four datasets:
1) [PACS](https://drive.google.com/file/d/13XXgVqJ2cVGGcL3afh3sbDQ6FV-O-oDw/view?usp=sharing)
2) [OfficeHome](https://drive.google.com/file/d/1eeafkGeLjxh4hduAcnyjT0Gr3FA7sxoj/view?usp=sharing)
3) [TerraINC](https://drive.google.com/file/d/1OQbbya0fDwwa-UyQe2VZG5_lgOyTdNe8/view?usp=sharing)
4) [MiniDomainNet](https://drive.google.com/file/d/1KMPXiRXh5SUTcQSWYmnRBQuD4MRGFSDX/view?usp=sharing)

Please download and extract them to `./data` directory or a directory of your own choosing and then change the `--dataroot` argument to that directoory.



