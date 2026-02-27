# GLAND: A Global and Local Attention‐Based Framework for Detecting Small and Non‐Continuous Spatial Domains

## Overview
GLAND is a deep learning framework for spatial domain identification that resolves multi-scale spatial dependencies by integrating global and local attention mechanisms. Using a novel spatial outlier detection strategy (spLOF) and a dual-branch encoder to simultaneously capture long-range biological similarities and localized spatial continuity, GLAND consistently improves clustering accuracy across 39 tissue slices from seven diverse spatial transcriptomic platforms. It also enables the high-resolution identification of non-continuous and micro-scale domains, effectively resolving fragmented tumor regions and small lymphoid aggregates within the complex tumor microenvironment.

## Installation

We recommend using a Conda environment to manage dependencies. You can install the core requirements directly from the repository:

```bash
# Clone the repository
git clone [https://github.com/CSUBioGroup/GLAND.git](https://github.com/CSUBioGroup/GLAND.git)
cd GLAND

# Install dependencies
pip install -r requirements.txt
```

For detailed instructions, including R environment configuration and GPU setup, please visit our [Installation Guide](https://gland.readthedocs.io/en/latest/installation.html).

## Compared Tools

The following state-of-the-art methods were used for performance benchmarking:

* [SpaGCN](https://github.com/jianhuig/SpaGCN)
* [CCST](https://github.com/xiaoyeye/CCST)
* [conST](https://github.com/ys-zong/conST)
* [DeepST](https://github.com/STOmics/DeepST)
* [ConST](https://github.com/rli012/ConST)
* [STAGATE](https://github.com/QIAO-SU/STAGATE)
* [SpaceFlow](https://github.com/hongleir/SpaceFlow)
* [GraphST](https://github.com/JinmiaoChenLab/GraphST)

## Download Data

The datasets used in this study can be downloaded from Zenodo:
👉 [https://zenodo.org/records/18756215](https://zenodo.org/records/18756215)


## License

This project is covered under the **MIT License**.

## Tutorial

For the step-by-step tutorial on data processing and spatial domain identification, please refer to our official documentation:
📖 [https://gland.readthedocs.io/en/latest/index.html#](https://gland.readthedocs.io/en/latest/index.html#)


## Contact

If you have any questions, please do not hesitate to contact us at:

📩 **Email:** [244712166@csu.edu.cn](mailto:244712166@csu.edu.cn)


## Cite










