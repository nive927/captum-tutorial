# Using Captum for Interpreting PyTorch Models in Production

This repository contains complete examples for using [Captum](https://github.com/pytorch/captum), an open-source interpretability library for PyTorch, to analyze, debug, and monitor movie recommendation models in production.

## Medium Blog Post

For a complete walkthrough with explanations, insights, and visuals: [**Using Captum for Interpreting PyTorch Models in Production**](https://medium.com/@nivedhithadm/using-captum-for-interpreting-pytorch-models-in-production-ddf91fbbd3fd)

## Repository Structure

- [captum_mlip_tutorial.ipynb](./captum_mlip_tutorial.ipynb): Main tutorial with robust interpretability metrics, embedding analysis, and production-ready insights.
- [captum_movielens_tutorial.ipynb](./captum_movielens_tutorial.ipynb): A minimal standalone version based on MovieLens for quick testing.
- [requirements.txt](./requirements.txt)`: List of dependencies.
- [data/](./data/): Folder containing processed datasets.
- [images/](./images/): Attribution plots and other visualizations.
- [docs/](./docs/): Supporting documentation and logs.


## Setup Instructions

```bash
git clone https://github.com/your-username/captum-movie-rec.git
cd captum-movie-rec
conda create -n captum_env python=3.10 -y
conda activate captum_env
pip install --upgrade pip
pip install -r requirements.txt
pip install <package-name>
```
