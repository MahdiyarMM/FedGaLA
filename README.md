# FedGaLA
Federated Unsupervised Domain Generalization Using Global and Local Alignment of Gradients

## Introduction
This is the official documentation for FedGaLA, submitted to ICML 2024. FedGaLA performs gradient alignment at both the local (client) and global (server) levels to address the problem of domain generalization for multi-domain data under privacy-preserving constraints.

## Requirements
To train one of the baselines, please follow the instructions below:

1. Install the requirements with the following command:
   `pip install -r requirements.txt`
2. Download the datasets from the corresponding links in the Data section and extract them to the `./data` folder.
3. Select the Self-Supervised Learning (SSL) method from `byol`, `simsiam`, `moco`, or `simclr`.
4. Run the following code to train the baseline model with the selected SSL method. The example below is for the FedSimCLR model on the PACS dataset:

```bash
python main.py --test_domain s --dataset pacs --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 7 --backbone resnet18 --aggregation FedAVG --SSL simclr --labeled_ratio 0.3 --workers 2
```

To train FedGaLA, run the following code:
```bash
python main.py --test_domain s --dataset pacs --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 7 --backbone resnet18 --aggregation GA --SSL simclr --labeled_ratio 0.3 --client_gm LA --local_threshold 0.0 --gamma 0.00 --workers 2
```
In the code above, `GA` stands for Global Alignment, and `LA` stands for Local Alignment. By default, the aggregation is set to FedAVG, and `client_gm` is set to None. For the `workers` parameter, please set it to `0` if you are using a Windows machine.

## Data
We train on the following four datasets:
1. [PACS](https://www.v7labs.com/open-datasets/pacs)
2. [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
3. [TerraINC](https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz)
4. [DomainNet](http://ai.bu.edu/M3SDA/)

For the PACS dataset, please download this [csv file](https://drive.google.com/file/d/19DZCyBbe_F_-7iUrTxG-AEDlpIUzvpFJ/view?usp=sharing) and place it in the designated directory.

Please download and extract them to the `./data` directory or a directory of your choosing, then change the `--dataroot` argument to that directory.



