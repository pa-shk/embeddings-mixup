# embeddings_mixup
A series of experiments on implementing MixUp augmentation technique for BERT architecture
## Data
All experiments were conducted on [rotten_tomatoes](http://https://huggingface.co/datasets/rotten_tomatoes "rotten_tomatoes") dataset. Dataset contains three splits - train (8.53k rows), validation (1.07k rows) and test (1.07k rows). This is a binary classification dataset for movie reviews with the equal proportion of both positive and negative reviews in each split. The distribution of the length of texts (train split) in tokens after applying bert-base-cased tokenizer can be seen below. ![image](https://github.com/pa-shk/embeddings_mixup/assets/108901776/c8369d09-6208-43b5-aa1e-0292f2b017c5)

Length distribution does not differ substantially across 3 splits of the dataset.
## Experiments
In all experiments I use bert-base-cased model with the following hyperparameters:
- max_len = 70
- batch_size = 256
- num_epochs = 10
- lr = 5e-5
- optimizer = AdamW
- scheduler = linear_schedule_with_warmup
- num_warmup_steps = 100

For evaluation, I've used mainly three metrics - accuracy, precision and recall. The following table provides quick information on all experiments, including short description of augmentation used, metrics on validation set, links to learning curves logged on wandb and commits.

| n | augmentation                                                                                                     | val_accuracy | val_precision | val_recall | wandb                                                                                         | commit  |
|---|------------------------------------------------------------------------------------------------------------------|--------------|---------------|------------|-----------------------------------------------------------------------------------------------|---------|
| 1 | baseline (no augmentation used)                                                                                  | 0.851        | 0.863         | 0.843      | [link](https://wandb.ai/pvlshknv/bert-base-cased-mixup/runs/fic488y2?workspace=user-pvlshknv) | 8dd0623 |
| 2 | MixUp applied to input embeddings                                                                                | 0.866        | 0.861         | 0.869      | [link](https://wandb.ai/pvlshknv/bert-base-cased-mixup/runs/rsvhtgxu?workspace=user-pvlshknv) | 456399a |
| 3 | MixUp applied to outputs of BERT pooler ([Mixup-Transformer](https://aclanthology.org/2020.coling-main.305.pdf)) | 0.851        | 0.861         | 0.844      | [link](https://wandb.ai/pvlshknv/bert-base-cased-mixup/runs/xamg7b8l?workspace=user-pvlshknv) | c88f8a9 |

To combine pairs of embeddings and labels I've used [standard MixUp](https://arxiv.org/abs/1710.09412) with slight modification in regard of the choice of the coefficient with which pairs are added together proposed in [article](https://math.mit.edu/research/highschool/primes/materials/2020/Zhao-Lialin-Rumshisky.pdf).

## Model weights availability
Model fine-tuned with augmentation (2) can be [downloaded](https://huggingface.co/pa-shk/bert-base-cased-embed-mixup) from Hugging Face hub.
inference.ipynb contains an example of running inference on the model.

## TODO
1. Experiment with placing MixUp after BERT encoder and before pooler
2. Implement [E-Stitchup MixUp](https://arxiv.org/pdf/1912.00772.pdf)
3. Implement [AttentionMix](https://arxiv.org/pdf/2309.11104.pdf)
