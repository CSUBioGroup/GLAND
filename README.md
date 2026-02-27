# GLAND: A Global and Local Attention‐Based Framework for Detecting Small and Non‐Continuous Spatial Domains

## Overview
Recent advances in spatial transcriptomic technology have provided unprecedented insights into tissue architecture by enabling high-throughput gene expression profiling with the corresponding spatial context. While identifying spatial domains is a prerequisite for understanding tissue heterogeneity and functional organization, current computational methods often fail to resolve non-continuous or small-scale domains that, despite their biological significance, exhibit complex spatial distribution. To address these challenges, we present GLAND, a deep learning framework that integrates global and local attention to resolve multi-scale spatial dependencies. GLAND employs a spatial outlier detection strategy to refine the spatial graph, coupled with a dual-branch encoder that simultaneously captures long-range biological similarity and localized spatial continuity. Comprehensive benchmarking across 39 slices and 7 spatial transcriptomic platforms demonstrates that GLAND exhibits superior performance in identifying not only traditional continuous spatial domains but also non-continuous and small domains. Within the complex tumor microenvironment, GLAND accurately resolves fragmented tumor regions and small lymphoid aggregates. Its learned spot embeddings bridge spatial gaps by preserving the transcriptomic identity of tumor domains across distal locations, facilitating a unified characterization of heterogeneous tissue landscapes.

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


## Contact / 联系

If you have any questions, please do not hesitate to contact us at:

📩 **Email:** [244712166@csu.edu.cn](mailto:244712166@csu.edu.cn)


## Cite









