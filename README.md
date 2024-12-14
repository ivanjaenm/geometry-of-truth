# CS 839 Final Project - The Multimodal Geometry of Truth

## Motivation

### Maths:
If the linear structures identified in these true/false datasets could also manifest in mathematical problems, they would significantly enhance mathematical reasoning. For instance, in tackling complex math problems, employing methods like chain-of-thought (CoT), tree-of-thought, and step-by-step approaches could improve reasoning accuracy. So, if geometric structures are integrated into the math reasoning process, models could autonomously verify the correctness of each step, perfectly complementing these techniques.

### Images:
Visual data often represents factual information (e.g., "The cat is black"). Investigating whether true/false distinctions can be linearly separable in the latent space of multi-modal LLMs helps us understand how models process visual truth. This has practical applications in detecting manipulated images or identifying false visual claims.

## Main Hypothesis
The original paper tests whether true and false statements for unambiguous text-based problems exhibit linear separability. One possible followup question is if LLMs develop a universal concept of truth, disentangled from specific languages. We hypothesize that a similar linear structure may emerge for true/false statements in math problems as well and potentially for classifications in image-based tasks.
Statement - The latent space of LLMs and multi-modal models (e.g., CLIP, GPT-4-math) exhibits linear structure for true/false distinctions in both images and math datasets.

## Methodology

### Dataset Preparation
#### Image Dataset:
Collect or create images paired with factual claims, labeled as true or false (e.g., "The sky is blue" vs. "The sky is green").
Ensure diversity by including real-world images, ambiguous claims, and doctored visuals.
#### Math Dataset:
Curate simple mathematical statements with clear truth values 
(e.g., "2 + 2 = 4" vs. "2 + 2 = 5").

### Embedding Generation
Use multi-modal models (such as LLaMA 3) to extract latent representations for image-text pairs and mathematical statements.

### Evaluation Metrics
To maintain consistency with the paper, we will use the following metrics for evaluation:
Apply linear classifiers (e.g., SVMs) to identify separability in the latent space.
Use dimensionality reduction techniques (e.g., PCA, t-SNE) to visualize the organization of true/false representations.
Classification Accuracy: Can a linear model reliably separate true/false representations?
Cross-Domain Analysis: Compare latent structures between images and mathematical statements to identify universal patterns.


# The Geometry of Truth

This repository is associated to the paper <a href="https://arxiv.org/abs/2310.06824">*The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*</a> by Samuel Marks and Max Tegmark. See also our <a href="https://saprmarks.github.io/geometry-of-truth/dataexplorer">interactive dataexplorer</a>.

(<a href="https://github.com/saprmarks/geometry-of-truth">View this page on github</a>.)

## Set-up

Navigate to the location that you want to clone this repo to, clone and enter the repo, and install requirements.
```
git clone git@github.com:saprmarks/geometry-of-truth.git
cd geometry-of-truth
pip install -r requirements.txt
```
Before doing anything, you'll need to generate activations for the datasets. You should have your own LLaMA weights stored on the machine where you cloned this repo. Put the absolute path for the directory containing your LLaMA weights in the file `config.ini`; Huggingface repos are also supported. 

Once that's done, you can generate the LLaMA activations for the datasets you'd like to work with with a command like
```
python generate_acts.py --model llama-2-13b --layers 8 10 12 --datasets cities neg_cities --device cuda:0
```
These activations will be stored in the acts directory. If you want to save activations for all layers, simply use `--layers -1`.

## Files
This directory contains the following files:
* `dataexplorer.ipynb`: for generating visualizations of the datasets. Code for reproducing figures in the text is included.
* `few_shot.py`: for implementing the calibrated 5-shot baseline.
* `generalization.ipynb`: for training probes on one dataset and checking generalization to another. Includes code for reproducing the generalization matrix in the text.
* `interventions.py`: for reproducing the causal intervention experiments from the text.
* `probes.py`: contains definitions of probe classes.
* `utils.py` and `visualization_utils.py`: utilities for managing datasets and producing visualizations. 


