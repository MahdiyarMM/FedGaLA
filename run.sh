python main.py --test_domain s --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 1 --aggregation AbyA --client_gm Delta --workers 0 --SSL MOcO
# python main.py --test_domain s --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 7 --wandb BYOL_base --workers 0 --SSL BYOL
# python main.py --test_domain s --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 7 --wandb SimSiam_fedswag --aggregation AbyA --client_gm Delta --workers 0 --SSL SimSiam
# python main.py --test_domain s --dataroot ./data/PACS --labeled_ratio 0.1 --communication_rounds 100 --client_epoch 7 --wandb SimSiam_base --workers 0 --SSL SimSiam
