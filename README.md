# Fed_DG
Federated Self-Supervised Domain Generalization 


Run the below code to train Fed_DG

Linux:

model delta:
```bash
python main.py  --test_domain s --dataroot ../data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 5 --client_gm Delta --model_save_path ./saved_models_delta
```

baseline:
```bash
python main.py  --test_domain s --dataroot ../data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 5 --model_save_path ./saved_models
```

Windows:

model delta AbyA:
```bash
python main.py  --test_domain s --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 7 --client_gm Delta --model_save_path ./saved_models_delta --workers 0 --aggregation AbyA --gamma 0
```

model delta:
```bash
python main.py  --test_domain s --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 5 --client_gm Delta --model_save_path ./saved_models_delta --workers 0
```

baseline:
```bash
python main.py  --test_domain s --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 5 --model_save_path ./saved_models --workers 0
```


[Results Google Sheet](https://docs.google.com/spreadsheets/d/19BgZnVh8LhMfkKWvfXLoEh6b4Lg-1QadeXq8Lno9rH0/edit?usp=sharing)
