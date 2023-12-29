# Visual Classification via Description from Large Language Models
## Sachit Menon, Carl Vondrick
## <span style="color:red">*ICLR 2023, Notable Top 5% (Oral)*</span>

[[Paper]](https://arxiv.org/pdf/2210.07183.pdf)
[[arXiv]](https://arxiv.org/abs/2210.07183)

## Approach


![[latent-points]](./figs/latent-points.png)

The standard vision-and-language model compares image embeddings (white dot) to word embeddings of the category name (colorful dots) in order to perform classification, as illustrated in (a). We instead query large language models to automatically build descriptors, and perform recognition by comparing to the category descriptors, as shown in (b).

## Usage

First install the dependencies.

Either manually:
```
conda install pytorch torchvision -c pytorch
conda install matplotlib torchmetrics -c conda-forge
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
```

Or using the provided `.yml` file.
```
conda env create -f classbydesc.yml
```

To reproduce accuracy results from the paper: edit the directories to match your local machine in `load.py` and set `hparams['dataset']` accordingly. Then simply run `python main.py`.

All hyperparameters can be modified in `load.py`.

To generate example decisions and explanations as well as contrast from the CLIP decision, use the `show_from_indices` function in `load.py` after having run `main.py`. Details forthcoming.

Example displaying the predictions that differ between baseline CLIP and our method:
```
show_from_indices(torch.where(descr_predictions != clip_predictions)[0], images, labels, descr_predictions, clip_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_similarity)
```

Example outputs:
![[figs]](./figs/explanations.png)

### Generating Your Own Descriptors
See `generate_descriptors.py`. If you have a list of classes, you can pass it to the `obtain_descriptors_and_save` function to save a json of descriptors. (You will need to add your OpenAI API token.)


## Citation
```
@article{menon2022visual,
  author    = {Menon, Sachit and Vondrick, Carl},
  title     = {Visual Classification via Description from Large Language Models},
  journal   = {ICLR},
  year      = {2023},
}
```