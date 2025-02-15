# FedGaLA
Federated Unsupervised Domain Generalization Using Global and Local Alignment of Gradients
<p align="center">
  <img src="assets/overview.png" width="550" title="hover text">
</p>

## Introduction
FedGaLA is a novel framework for federated unsupervised domain generalization, combining global and local gradient alignment to enhance model generalization across diverse domains while preserving privacy. This repository contains the official implementation of FedGaLA as described in our paper published in *AAAI 2025*.

## Features
- **Federated Learning Framework**: Incorporates both server-level (global) and client-level (local) gradient alignment techniques.
- **Domain Generalization**: Designed to handle multi-domain datasets under privacy constraints.
- **Flexible Self-Supervised Learning (SSL) Backbones**: Supports `byol`, `simsiam`, `moco`, and `simclr` SSL methods.
- **Ease of Use**: Simple and modular implementation for easy experimentation and adaptation.

For the full paper see [here](https://arxiv.org/pdf/2405.16304).

---

## Requirements
Follow these steps to set up your environment and train models:

1. **Install Dependencies**  
   Use the following command to install all required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2.	**Download Datasets**
Use the links provided in the Data section. Extract the datasets into the ./data folder (or a directory of your choice).
3.	**Choose an SSL Method**
Select a Self-Supervised Learning method from byol, simsiam, moco, or simclr.
4.	**Train Baseline Models**
To train a baseline model (e.g., FedSimCLR on the PACS dataset), use the following command:

```bash
python main.py --test_domain s --dataset pacs --dataroot ./data/PACS --labeled_ratio 0.1 \
--communication_rounds 100 --client_epoch 7 --backbone resnet18 --aggregation FedAVG \
--SSL simclr --workers 2
```

5.	**Train FedGaLA**
Use the following command to train the FedGaLA model:

```bash
python main.py --test_domain s --dataset pacs --dataroot ./data/PACS --labeled_ratio 0.1 \
--communication_rounds 100 --client_epoch 7 --backbone resnet18 --aggregation GA \
--SSL simclr --client_gm LA --local_threshold 0.0 --gamma 0.00 --workers 2
```

## Parameters:
- `GA`: Global Alignment (server-level).
- `LA`: Local Alignment (client-level).
- `Default aggregation`: FedAVG (if aggregation is not specified).
- `--workers` 0 (For Windows users)

## Data

FedGaLA supports the following datasets:
1. [PACS](https://www.v7labs.com/open-datasets/pacs)
2. [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
3. [TerraINC](https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz)
4. [DomainNet](http://ai.bu.edu/M3SDA/)

For the PACS dataset, please download this [csv file](https://drive.google.com/file/d/19DZCyBbe_F_-7iUrTxG-AEDlpIUzvpFJ/view?usp=sharing) and place it in the designated directory.

Please download and extract them to the `./data` directory or a directory you choose, then change the `--dataroot` argument to that directory.


## Citation

If you find this work useful, please consider citing:

```bash
@article{FedGaLA2025,
  title={Federated Unsupervised Domain Generalization Using Global and Local Alignment of Gradients},
  author={Pourpanah, Farhad and Molahasani, Mahdiyar and Soltany, Milad and Greenspan, Michael and Etemad, Ali},
  journal={AAAI},
  year={2025}
}
```
```bash
@article{FedDG2024,
  title={A Theoretical Framework for Federated Domain Generalization with Gradient Alignment},
  author={Molahasani, Mahdiyar and Soltany, Milad and Pourpanah, Farhad and Greenspan, Michael and Etemad, Ali},
  journal={NeurIPS Workshop on Mathematics of Modern Machine Learning},
  year={2024}
}
```
## License

This project is licensed under the MIT License.

## Our team

- [Mahdiyar Molahasani](https://github.com/MahdiyarMM) 
- [Milad Soltany](https://github.com/miladsoltany) 
- [Farhad Pourpanah](https://github.com/Farhad0086) 

Feel free to check out our GitHub profiles for more of our work!

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, features, or bug fixes.

## Contact

For questions or inquiries, please reach out to m.molahasani@queensu.ca.

Happy experimenting with FedGaLA!



