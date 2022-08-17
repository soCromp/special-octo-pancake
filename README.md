# special-octo-pancake

Setup
- Install conda environment to run CLIP using `conda env create -f conda_autoLF.yml` and `conda activate autolf`
- Run `python -m spacy download en_core_web_md` to download model for keyword extraction
- Obtain the waterbirds dataset and update DIR in infer.py and remspur.py
- Download ClipCap checkpoint from [https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view) and update model_path in infer.py

Use [run.sh](https://github.com/soCromp/special-octo-pancake/blob/main/run.sh) to execute the following steps automatically:

Steps
1. [`infer.py`](https://github.com/soCromp/special-octo-pancake/blob/main/infer.py) generates keywords for images from automatic captions and for labels from Wikipedia articles, then determines which caption keywords are spurious and outputs them to spurious.txt, and 
2. [`remspur.py`](https://github.com/soCromp/special-octo-pancake/blob/main/remspur.py) reads in the spurious words, then applies autoLF to the images to black out image patches in which CLIP predicts a spurious word (for a notebook walkthrough version, go to [remspur.ipynb](https://github.com/soCromp/special-octo-pancake/blob/main/remspur.ipynb). In order, the command line arguments are comma-separated list of different thresholds, pixel length of the side of a patch and number of spurious words, eg `python remspur.py 0.7 100 5`.

You only need to run infer.py once for a given dataset, while remspur.py can be run for different hyperparameter settings (threshold, patch size, number of spurious words).

Note: remspur.py processes *all* the images; make sure when you use these images downstream that you still use an unaltered test set.
