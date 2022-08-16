# special-octo-pancake

Setup
- Install conda environment to run CLIP using `conda env create -f conda_autoLF.yml` and `conda activate autolf`
- Run `python -m spacy download en_core_web_md` to download model for keyword extraction
- Obtain the waterbirds dataset and update DIR in infer.py and remspur.py
- Download ClipCap checkpoint from [https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view) and update model_path in infer.py

Use [run.sh](https://github.com/soCromp/special-octo-pancake/blob/main/run.sh) to execute all the following steps automatically:

Steps
1. [`infer.py`](https://github.com/soCromp/special-octo-pancake/blob/main/infer.py) generates captions for images and extracts these captions' keywords,
2. [labelkeys.py](https://github.com/soCromp/special-octo-pancake/blob/main/labelkeys.py) gets keywords for label names, then determines which caption keywords are spurious and outputs them to spurious.txt,
3. [remspur.py](https://github.com/soCromp/special-octo-pancake/blob/main/remspur.py) reads in the spurious words, then applies autoLF to the images to black out image patches in which CLIP predicts a spurious word (for a notebook walkthrough version, go to [remspur.ipynb](https://github.com/soCromp/special-octo-pancake/blob/main/remspur.ipynb)) and
4. [filemmt.py](https://github.com/soCromp/special-octo-pancake/blob/main/filemmt.py) ensures that the file structure works with other parts of the code.

Note: this processes *all* the images; make sure when you use these images downstream that you still use an unaltered test set.
