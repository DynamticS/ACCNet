# ACCNet: Adaptive Cross-frequency Coupling Graph Attention for EEG Emotion Recognition



## DATASET

Download the Processed FACED dataset and extract all files to path : ./data/raw_EEG.

## Run

After configuring the environment according to the requirements.txt, execute:

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --T 0.7 --model 'ACCNet' --label-type 'NT' --edge-compute 'COS' --learning-rate 1e-3 --GNN-inheads 4 --data-prepare True
```

