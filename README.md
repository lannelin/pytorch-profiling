# Pytorch Profiling

By example profiling of pytorch training.

Goal: train a ViT on Flowers102 dataset, diagnosing situations where we are IO bound, CPU-bound, other


## Snippets for 25/11 Training

### Setup interactive session

```bash
# on baskerville
srun --export=USER,HOME,PATH,TERM --account vjgo8416-hpc2511 --qos=turing --nodes=1-1 --cpus-per-gpu=36 --reservation vjgo8416-hpc2511 --gres=gpu:1 --time=1:0:0 --pty /bin/bash
# wait...

```bash
module load Python/3.11.3-GCCcore-12.3.0

# for convenience
export PROJECT=vjgo8416-hpc2511
export PROJECT_DIR=/bask/projects/v/vjgo8416-hpc2511
export PROFILING_SHARED=$PROJECT_DIR/profiling_shared

# activate pyuthon env
source ${PROFILING_SHARED}/env/bin/activate
cd ${PROJECT_DIR}/$USER/hpc-training-nov-2025/4-Profiling

# share pre-downloaded models
export TORCH_HOME=${PROFILING_SHARED}/torch_cache
```

## Overlapping an interactive session

```bash
squeue # get JOB ID
srun --pty --overlap --jobid <YOUR_JOBID> bash
```


## Nvidia-SMI

Simple output on loop:
`nvidia-smi -l 1`

Csv output on loop:

monitor gpu 0 for utilisation+memory every 1 second, include timestamp and print delimiter-friendly
`nvidia-smi dmon -i 0 -s mu -d 1 -o TD`


## Train a model

```bash
python runner.py fit -c configs/training.yaml --trainer.logger.name test --trainer.max_epochs 2
```

to train a cpu model for 1 epoch (slow):
```bash
python runner.py fit -c configs/training.yaml --trainer.logger.name cpu --trainer.max_epochs 1 --trainer.accelerator cpu
```

## Tensorboard
```bash
port=<my_port>
tensorboard --port ${port} --host $(hostname).cluster.baskerville.ac.uk --logdir $(pwd)/logs
```

## Mixed precision

### manual inference

check we have a checkpoint:
```bash
ls -lh logs/test/version_0/checkpoints/epoch\=1-step\=64.ckpt
```

```python
import os

from PIL import Image
from pytorch_profiling.example.vit import ViTB16
import torch
from torchvision import transforms

impath = os.path.join(os.environ["PROFILING_SHARED"], "data", "Blanket_99.jpg")
im = Image.open(impath)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

device="cuda:0"
model = ViTB16.load_from_checkpoint("logs/test/version_0/checkpoints/epoch=1-step=64.ckpt", num_classes=102).to(device)
model.eval()
ims = transform(im).unsqueeze(0).to(device) # transform data, unsqueeze, put on device

# uncomment below for bfloat16
#model = model.bfloat16() # model to bfloat16
#ims = ims.bfloat16() # convert data to bfloat16

with torch.inference_mode():
    output = model(ims)

print(output.argmax(dim=1))
```

### lightning training

observe differences in config
```bash
diff configs/training_bf16.yaml configs/training.yaml
```

```bash
python runner.py fit -c configs/training_bf16.yaml --trainer.logger.name bf16 --trainer.max_epochs 2
```

## Torch's `inference_mode`


without `inference_mode`:
```python
import torch
from pytorch_profiling.example.vit import ViTB16

device="cuda:0"
n_images = 2
model = ViTB16(num_classes=102, freeze_embedding=True).to(device)
model.eval()
ims = torch.randn(n_images, 3, 224, 224).to(device)

output = model(ims)
print(output.argmax(dim=1)) # meaningless
```

again but with `inference_mode`:
```python
import torch
from pytorch_profiling.example.vit import ViTB16

device="cuda:0"
n_images = 2
model = ViTB16(num_classes=102, freeze_embedding=True).to(device)
model.eval()
ims = torch.randn(n_images, 3, 224, 224).to(device)

with torch.inference_mode():
    output = model(ims)

print(output.argmax(dim=1)) # meaningless
```


## DataLoader::num_workers

```bash
python runner.py fit -c configs/training.yaml –c configs/sleepy_data.yaml  --trainer.logger.name 0workers --trainer.max_epochs 2  --data.num_workers 0
python runner.py fit -c configs/training.yaml –c configs/sleepy_data.yaml  --trainer.logger.name 1workers --trainer.max_epochs 2  --data.num_workers 1
python runner.py fit -c configs/training.yaml –c configs/sleepy_data.yaml  --trainer.logger.name 2workers --trainer.max_epochs 2  --data.num_workers 2
python runner.py fit -c configs/training.yaml –c configs/sleepy_data.yaml  --trainer.logger.name 16workers --trainer.max_epochs 2  --data.num_workers 16
```



## Other


useful: https://huggingface.co/spaces/hf-accelerate/model-memory-usage
