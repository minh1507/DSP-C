from src.01_load_dataset import extract_dataset, load_samples
from src.05_train import train_model
from src.06_infer import inference

# 1. Tải & load dataset
extract_dir = extract_dataset()
samples = load_samples(extract_dir)

# 2. Train GNN
model, val_loader = train_model(samples, epochs=10, batch_size=16)

# 3. Inference trên validation set
val_samples = [s for s in samples[int(0.8*len(samples)) : ]]
preds = inference(model, val_samples)

for s, p in zip(val_samples, preds):
    print(f"{s['id']}: True={s['label']}, Pred={p}")
