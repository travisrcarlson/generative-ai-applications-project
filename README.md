# Generating Images of Surface Landmines

A class-conditional generative adversarial network (GAN) in PyTorch that generates 64 × 64 colour images of two kinds of surface landmine, the PFM-1 and the PMA-2, on request. It learns from crops cut around the labelled items in SULAND, a public set of real video frames of mine surrogates in grass, leaves and gravel, and its images are judged against real crops: by eye, by a check for copies of training crops, by training a classifier on them and testing it on real photos at the training site and at a new one, and by a blind review by an EOD specialist.

Generated mine images could serve as training images for a model that finds mines in drone or survey photos, for recognition training and awareness material, and for digital worlds in which drones, or drone operators, train. A wrong shape matters more than a wrong colour, since mines can be painted to camouflage, and a picture that teaches people what a mine looks like has to be grounded in engineering data and be as accurate as possible. These images are not, so they are judged here as training material, not as reference pictures. This is the fifth project of my AI capstone on humanitarian mine action.

## Result
The generator learns both types without copying its training crops. A classifier trained on real crops recognises 91% of the generated PFM-1s and 75% of the generated PMA-2s as their type, about as often as it recognises real test crops (93% and 73%). But the images do not stand in for real ones. A classifier trained on generated mines scores 0.99 macro-F1 on generated test images, yet it finds 3% of the real mines at the training site and almost none at the US site. The same classifier trained on real crops finds 84% and 38%. Trained on generated mines beside real ground, it learns to tell generated images from real photos, not mines from ground. In a blind review of 32 crops, I told generated from real in 30 of them. I found 12 of the 16 generated crops too indistinct to identify, and judged the shape right in only 4 of them, none a PFM-1. These images are not fit to show anyone what a mine looks like.

## What's in this repository
- `generative_model.ipynb`: the task and the data; checks of the data files, the frames, the labels and data quality; how the crops were made; the generator and discriminator and their training; generated samples; and the evaluation: a copy check, the classifier check at two sites and why the classifier trained on generated images fails, what a classifier trained on real crops makes of the samples, images between the two types, and a blind review; and a summary.
- `data/mine_crops_64.zip`: 10,397 crops of the two mine types, 64 × 64 pixels, from Italy's training and test splits and the US set, made from the dataset's frames by the code in section 3.1 of the notebook.
- `data/background_crops_64.zip`: 9,911 squares of the scene with no mine in them, one per frame used, for the classifier check.
- `data/crops_manifest.csv`: one row per crop, with its split, type, frame, video, box and square.
- `data/frames.csv` and `data/boxes.csv`: every frame and every labelled box of the dataset's four splits, for the data profile.
- `data/sample_photos/`: six of the dataset's frames, shrunk to 800 pixels on the longer side, for the notebook's first figure.
- `figures/`: the notebook's figures.
- `Generative_AI_Analysis_Report.pdf` and `module_summary.pdf`: the report. The two files are identical; the brief uses both names.
- `requirements.txt`: the Python environment, from `pip freeze`.

## Dataset
SULAND (SUrface LANDmine), by Emanuele Vivoli, Marco Bertini and Lorenzo Capineri of the Media Integration and Communication Center, University of Florence (https://github.com/miccunifi/SULAND-Dataset; Vivoli et al., 2024, *Remote Sensing*, 16(4), 677, https://doi.org/10.3390/rs16040677), with the refined SULAND_v2 labels by Sagar Lekhak, Prasanna Reddy Pulakurthi, Lalit Joshi, Ramesh Bhatta and Emmett J. Ientilucci of the Rochester Institute of Technology (https://arxiv.org/abs/2607.28996), from the Hugging Face record https://huggingface.co/datasets/SagarLekhak/SULAND_v2_RGB_Surface_Landmine_Dataset at revision `36f054cd593804decac44666a7c721b2106b7cff`.

- Contents: 33,771 video frames, 1440 × 1920 pixels, from 55 videos taken with a phone camera of surrogates of the PFM-1 and the PMA-2: Italy's training (22,756), validation (2,836) and test (3,743) splits, made by video, and a US set (4,436) from another site; 12,427 labelled boxes.
- Files used: `SULAND_images.zip` (16,674,324,895 bytes) and `Annotation_files_for_SULAND_v2.zip` (7,412,917 bytes), read on 2026-10-21. The frames were read from the first by HTTP range requests, about 5 GB in all, each checked against the archive's CRC-32, so the archive is not downloaded whole and is not in this repository.
- Crops: the notebook's section 3.1 holds the code that made them. Set `REBUILD_CROPS = True` in section 1.1 and run the notebook, and that section reads the frames again and rebuilds the crops into `rebuilt_crops/`, checking that they are identical to the shipped files. The notebook checks the shipped files' SHA-256 checksums when it starts:
  - `mine_crops_64.zip`: `2ddc12cd38dac2b2d621341171b3163d6e7b74d018854b09edabc7cceaa49adb`
  - `background_crops_64.zip`: `9cfe946399dfae19f4d7a9be82e9cdca93ab0d03c7fb933e9d7f4a36b2326f15`
  - `crops_manifest.csv`: `cac0c5593c0b37df522bc7ac3762bf1801d0edf11f5148760c37dbecfe69c286`
  - `frames.csv`: `6001c0f7f49ff9ef71db3547fb51791369d08100226f070430b8db6bf624ba5d`
  - `boxes.csv`: `5e82e9d8bbfaca2f76afe5f2f9b3aab5dc43a6d4a0207827f80aec72e365ad59`
- Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), for both the original release (MICC, 2023) and the SULAND_v2 labels. The crops, the tables and the sample frames are adapted from the dataset under that licence: the items were cut out of the frames and resized, the labels were turned into tables, and the sample frames were shrunk. The record states that the dataset "is intended for academic research".
- Use: the items are surrogates of two mine types, filmed at two sites, so the generator can only make images of these two surrogates in these kinds of places. Every image it makes is labelled as generated, and none should be used to show people what a mine looks like unless an expert has checked it.

## How to run
Needs Python 3.13 and Git. `requirements.txt` was generated on Python 3.13.0 with the CPU build of PyTorch, which comes from PyTorch's own package index; the install line below adds that index.

```
git clone https://github.com/travisrcarlson/generative-ai-applications-project.git
cd generative-ai-applications-project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
jupyter notebook generative_model.ipynb
```

On macOS or Linux, activate with `source .venv/bin/activate`; on macOS, also change `torch==2.14.1+cpu` in `requirements.txt` to `torch==2.14.1` before installing, since PyTorch publishes no `+cpu` build for macOS. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. In Jupyter, choose **Kernel > Restart Kernel and Run All Cells**.

A full run takes about half an hour on a two-core machine (27 and 31 minutes in two runs): training the GAN takes 15 to 18 minutes of it and the classifier checks most of the rest, and the notebook prints how long each part took. Training runs on the CPU with two threads, fixed seeds and deterministic algorithms, so a rerun on the same computer gives the same images and scores. On another computer the generated images and the scores can differ somewhat, because training amplifies small differences in how processors round sums; the blind review in section 7.5 refers to the sheet in `figures/10_blind_review.jpg`.
