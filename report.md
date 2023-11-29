
## Report
    * repository: https://github.com/XiaopeiYang/classify_by_description_release

   * We use torchvision to download the dataset and change the file path to execute the program.

   * For achitecture ViT-L/14, we change the batch size = 64*5; For architecture ViIT-L/14@336px, we change the batch size= 64. The reason is that we found the orginal batch size= 64*10 will cause the out of usable memory problem.

   * The dataset contains EuroSAT, Places365, Food101, Oxford Pets and Describable Textures. And we show the results as graphs below. 






## EuroSAT
>|Architecture for $\phi$   | Ours | CLIP | $\Delta$ |
>|-----------------------|------|--------|--------|
>| ViT-B/32              | 48.49 | 44.28 | 4.21   |
>| ViT-B/16              | 50.61 | 46.19 | 4.42   |
>| ViT-L/14              | 47.33 | 36.94 | 10.39 |
>| ViT-L/14@336px        | 47.02 | 38.11 | 8.91 |


## Places 365







## Food101

>|Architecture for $\phi$   | Ours | CLIP | $\Delta$ |
>|-----------------------|------|--------|--------|
>| ViT-B/32              | 48.49 | 44.28 | 4.21   |
>| ViT-B/16              | 50.61 | 46.19 | 4.42   |
>| ViT-L/14              | 47.33 | 36.94 | 10.39 |
>| ViT-L/14@336px        | 47.02 | 38.11 | 8.91 |


## Oxdord Pets




## Describable Textures

>|Architecture for $\phi$   | Ours | CLIP | $\Delta$ |
>|-----------------------|------|--------|--------|
>| ViT-B/32              | 42.45 | 41.49 | 0.96   |
>| ViT-B/16              | 50.61 | 46.19 | 4.42   |
>| ViT-L/14              | 55.05 | 50.85 | 4.2    |
>| ViT-L/14@336px        | 55.00 | 51.06 | 3.94   |
