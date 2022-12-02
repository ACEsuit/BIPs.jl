# BIPs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cortner.github.io/BIPs.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cortner.github.io/BIPs.jl/dev/)
[![arXiv](https://img.shields.io/badge/arXiv-2207.08272-00ff00.svg)](https://arxiv.org/abs/2207.08272)


This is the official implementation of the *Boost Invariant Polynomial (BIP) for Efficient Jet Tagging* proposed [here](https://arxiv.org/abs/2207.08272).

## Installation

`BIPs.jl` is registered in the `MolSim` registry, which can be installed from the Julia REPL, package manager mode `]` via 
``` 
registry add https://github.com/JuliaMolSim/MolSim.git
```
Then the package can be installed by simply call (still in the package manager)
``` 
add BIPs
```

Official Implementation of the paper: **BIP: Boost Invariant Polynomials for Efficient Jet Tagging** [see here](https://inspirehep.net/literature/2116058)

Deep Learning approaches are becoming the go-to methods for data analysis in High Energy Physics (HEP). Nonetheless, most physics-inspired modern architectures are computationally inefficient and lack interpretability. This is especially the case with jet tagging algorithms, where computational efficiency is crucial considering the large amounts of data produced by modern particle detectors. In this work, we present a novel, versatile and transparent framework for jet representation; invariant to Lorentz group boosts, which achieves high accuracy on jet tagging benchmarks while being orders of magnitudes faster to train and evaluate than other modern approaches for both supervised and unsupervised schemes.


---
## Data 

  The data for this environment is going to be downloaded from the [Top Quark Tagging Reference Dataset](https://zenodo.org/record/2603256)

--- 
## Authors
👤 **Christoph Ortner**,  (UBC University) <br>
👤 **Ilyes Batatia**, (University of Cambridge)<br>
👤 **Jose M Munoz**, (Mitacs Intern)<br>
