# Team Ohio State at CMCL 2021 Shared Task

## Introduction
This is the code repository for the paper Team Ohio State at CMCL 2021 Shared Task: Fine-Tuned RoBERTa for Eye-Tracking Data Prediction.

Major dependencies include:

- Python 3.9+
- PyTorch 1.7.1+
- Pandas 1.2.2+
- TensorBoard 2.4.1+
- Transformers 4.3.2+

# Training
`main.py` is the main training script. A sample command for training the model is:
```
python main.py train --train_path training_data/training_data.csv \
                     --dep_var nfix \
                     --num_train_epochs 32 \
                     --batch_size 4 \
                     --learning_rate 5e-5
                     --pretrained_model roberta-large
```

# Inference
`main.py` is also the main inference script. A sample command for generating model predictions is:
```
python main.py test --test_path test_data/test_data.csv \
                    --dep_var nfix \
                    --pretrained_model roberta-large \
                    --checkpoint output/roberta-large_nfix_epoch32_batch4_lr5e-05_0/model.pth
```

Descriptions of other arguments for hyperparameter tuning and feature ablation are available in `model_args.py`.

# Evaluation
`get_mae.py` can be used to calculate the MAE for each eye-tracking feature:
```
python get_mae.py my_predictions.txt truth.txt
```

# Questions
For questions or concerns, please contact Byung-Doh Oh ([oh.531@osu.edu](oh.531@osu.edu)).