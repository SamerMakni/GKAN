# GKAN

This project implements Kolmogorov-Arnold Network on Graph structured data, I tried to imitate GCN architecture.

The `example.ipynb` notebook has an example implementation on the Cora dataset. 

## Requirements

This project was tested on Python 3.10.3. Follow the steps below to install dependencies:

- Create a new conda environment:

```bash
conda create -n gkan121 python=3.10
```

- Install Pytorch for the CUDA version at hand (in our case it is CUDA 12.2):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- Install Pytorch Geometric with optional dependencies

```bash
pip install torch_geometric

# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

- Install remaining dependencies:

```bash
pip install -r requirements.txt
```
