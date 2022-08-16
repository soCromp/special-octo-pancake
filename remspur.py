# Given a set of images and list of spurious words, remove patches from each image that correspond to a spurious word
import pandas as pd
import os
# from PIL import Image
from torchvision import io, transforms
from torchvision.utils import Image, ImageDraw, save_image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import torch
import clip
import sys

######## Settings ########
# Where the list of spurious words output by labelkeys.py is located
inpath = 'spurious.txt'
# Where your copy of wildbirds is located
DIR = './waterbirds_v1.0'
# Directory where you want the modified waterbirds images to go
outpath = './out'

thr = sys.argv[1] # thresholds before a patch is blacked out in form of comma-separated list eg 0.4,0.5,0.7
PATCH_SIZE = int(sys.argv[2]) # patch size
spurlen = int(sys.argv[3]) # number of spurious words to use

THRESH=[float(t) for t in thr.split(',')]
print('threshold:', THRESH, 'patch size:', PATCH_SIZE, 'spurious words length:', spurlen)

outdir = []
for thr in THRESH:
    outdir.append(os.path.join(outpath, f'th{thr}ps{PATCH_SIZE}wc{spurlen}'))
    if not(os.path.isdir(outdir[-1])):
            os.mkdir(outdir[-1])

with open(inpath, 'r') as f:
    spur = f.readlines()
spur = [l.strip() for l in spur]
spur = spur[:spurlen]
print(spur)

with open(os.path.join(DIR, 'metadata.csv'),'r') as f:
    meta = f.readlines()
meta = meta[1:]
names = [l.split(',')[1] for l in meta]

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in spur]).to(device)
with torch.no_grad(): 
    text_features = model.encode_text(text_inputs)

i=0
for fname in names:
    img = io.read_image(os.path.join(DIR, fname)).to(torch.float)
    img = torch.reshape(img, (1,img.shape[0],img.shape[1],img.shape[2]))
    kh, kw = PATCH_SIZE, PATCH_SIZE  # kernel size
    dh, dw = PATCH_SIZE, PATCH_SIZE  # stride
    patches = img.unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches.size()
    patches = patches.contiguous().view(patches.size(0), 3, -1, kh, kw)

    prepped = []
    for p in range(patches.shape[2]): 
        patch = patches[:,:,p,:,:].squeeze()
        prepped.append(preprocess(to_pil_image(patch)).unsqueeze(0).to(device))
    image_input = torch.cat(prepped).to(torch.uint8)

    with torch.no_grad(): 
        image_features = model.encode_image(image_input)

    # Pick the most similar label for the patch
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    sim, ind = similarity.max(dim=1) #for each patch, the strength with and index of the word of its best match
    
    patches_t = [patches]*len(THRESH)
    for t,thr,od in zip(patches_t, THRESH, outdir):
        t[:,:,sim>=thr] = torch.zeros(PATCH_SIZE, PATCH_SIZE)

        # Reshape back
        patches_orig = t.view(unfold_shape)
        output_h = unfold_shape[2] * unfold_shape[4]
        output_w = unfold_shape[3] * unfold_shape[5]
        patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_orig = patches_orig.view(1, 3, output_h, output_w)
        result = to_pil_image(patches_orig.to(torch.uint8).squeeze())

        makepath = os.path.join(od, fname.split('/')[0])
        if not(os.path.isdir(makepath)):
            os.mkdir(makepath)
        savepath = os.path.join(od, fname)
        result.save(savepath)

    i=i+1
    if i%1000 == 0: print(i/len(names), '% out')

