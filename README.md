# GLAND

GLAND is a spatial transcriptomics analysis tool designed to decipher spatial domains and cell-cell interactions.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

You can also install the package using `setup.py`:

```bash
python setup.py install
```

## Usage

Here is a basic example of how to use GLAND:

```python
import scanpy as sc
from GLAND import GLAND

# Load your data
adata = sc.read_h5ad('path/to/your/data.h5ad')

# Initialize and train the model
model = GLAND(adata)
adata = model.train()

# The result is stored in adata.obsm['emb']
print(adata.obsm['emb'])
```

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- Scanpy
- Faiss
