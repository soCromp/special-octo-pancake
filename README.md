# special-octo-pancake

Steps
1. infer.py generates captions for images and extracts these captions' keywords,
2. labelkeys.py gets keywords for label names, then determines which caption keywords are spurious and outputs them to spurious.txt and
3. remspur.py reads in the spurious words, then applies autoLF to the images to black out image patches in which CLIP predicts a spurious word.

Setup
- Obtain the waterbirds dataset and update DIR in infer.py
- Download ClipCap checkpoint from https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view and update model_path in infer.py
