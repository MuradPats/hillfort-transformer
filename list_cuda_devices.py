import torch

# Get torch version
print("Torch version:", torch.__version__)

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(i, props.name, f"{props.total_memory // 1024**2} MB")
