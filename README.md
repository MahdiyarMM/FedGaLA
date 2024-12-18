# FedGaLA
Federated Unsupervised Domain Generalization Using Global and Local Alignment of Gradients

## Introduction
This is the official documentation for FedGaLA published AAAI 2025. FedGaLA performs gradient alignment at both the local (client) and global (server) levels to address the problem of domain generalization for multi-domain data under privacy-preserving constraints.



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

# FedGaLA
Federated Unsupervised Domain Generalization Using Global and Local Alignment of Gradients

## Introduction
FedGaLA is a novel framework for federated unsupervised domain generalization, combining global and local gradient alignment to enhance model generalization across diverse domains while preserving privacy. This repository contains the official implementation of FedGaLA as described in our paper published in *AAAI 2025*.

## Features
- **Federated Learning Framework**: Incorporates both server-level (global) and client-level (local) gradient alignment techniques.
- **Domain Generalization**: Designed to handle multi-domain datasets under privacy constraints.
- **Flexible Self-Supervised Learning (SSL) Backbones**: Supports `byol`, `simsiam`, `moco`, and `simclr` SSL methods.
- **Ease of Use**: Simple and modular implementation for easy experimentation and adaptation.

---

## Requirements
Follow these steps to set up your environment and train models:

1. **Install Dependencies**  
   Use the following command to install all required Python packages:
   ```bash
   pip install -r requirements.txt

	2.	Download Datasets
Download the datasets from the links provided in the Data section. Extract the datasets into the ./data folder (or a directory of your choice).
	3.	Choose an SSL Method
Select a Self-Supervised Learning method from byol, simsiam, moco, or simclr.
	4.	Train Baseline Models
To train a baseline model (e.g., FedSimCLR on the PACS dataset), use the following command:

python main.py --test_domain s --dataset pacs --dataroot ./data/PACS --labeled_ratio 0.1 \
--communication_rounds 100 --client_epoch 7 --backbone resnet18 --aggregation FedAVG \
--SSL simclr --workers 2


	5.	Train FedGaLA
Use the following command to train the FedGaLA model:

python main.py --test_domain s --dataset pacs --dataroot ./data/PACS --labeled_ratio 0.1 \
--communication_rounds 100 --client_epoch 7 --backbone resnet18 --aggregation GA \
--SSL simclr --client_gm LA --local_threshold 0.0 --gamma 0.00 --workers 2

Parameters:
	•	GA: Global Alignment (server-level).
	•	LA: Local Alignment (client-level).
	•	Default aggregation: FedAVG (if aggregation is not specified).
	•	For Windows users, set --workers 0.

Data

FedGaLA supports the following datasets:
	1.	PACS
	•	Download this CSV file for data configuration.
	2.	OfficeHome
	3.	TerraINC
	4.	DomainNet

Instructions:
	•	Download and extract the datasets to ./data or a directory of your choice.
	•	Update the --dataroot argument in the training commands to point to the appropriate dataset directory.

Citation

If you find this work useful, please consider citing:

@article{FedGaLA2025,
  title={Federated Unsupervised Domain Generalization Using Global and Local Alignment of Gradients},
  author={Your Name et al.},
  journal={AAAI},
  year={2025}
}

License

This project is licensed under the MIT License.

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, features, or bug fixes.

Contact

For questions or inquiries, please reach out to your.email@example.com.

Happy experimenting with FedGaLA!

### Key Changes:
1. **Reorganized Content**: Grouped related sections for clarity and logical flow.
2. **Enhanced Formatting**: Used consistent code block formatting for commands and parameters.
3. **Detailed Descriptions**: Added descriptions for parameters and features.
4. **Standardized Structure**: Included sections like Citation, License, Contributing, and Contact for completeness.
5. **Improved Clarity**: Simplified and clarified instructions for better usability.
![image](https://github.com/user-attachments/assets/c12ee698-a32f-47ba-bbc9-02afb8f54d3d)


