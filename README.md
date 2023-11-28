# Fed_DG
Federated Self-Supervised Domain Generalization 


Run the below code to train Fed_DG
model delta:
    ```bash
    python main.py  --test_domain s --dataroot ../data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 5 --client_gm Delta --model_save_path ./saved_models_delta
   ```

baseline:
    ```bash
    python main.py  --test_domain s --dataroot ../data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 5 --model_save_path ./saved_models
   ```
