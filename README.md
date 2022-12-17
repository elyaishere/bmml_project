# Sharpness-Aware Minimization
project for Bayesian Methods of Machine Learning course 2022 @ Skoltech

# Evaluation results
[report](report.pdf)
### Cifar10

||epoch|top-1|top-5|
|:---|:---:|:---:|:---:|
|**SAM**|`100`<br>`200`<br>`400`|`14.5`<br>`12.48`<br>`11.07`|`0.7`<br>`0.55`<br>`1.08`|
|**No SAM**|`100`<br>`200`<br>`400`|`15.34`<br>`12.94`<br>`11.24`|`0.91`<br>`0.89`<br>`0.8`|

### Cifar100

||epoch|top-1|top-5|
|:---|:---:|:---:|:---:|
|**SAM**|`100`<br>`200`<br>`400`|`57.6`<br>`55.25`<br>`78.09`|`25.5`<br>`24.93`<br>`43.27`|
|**No SAM**|`100`<br>`200`<br>`400`|`89.93`<br>`90.19`<br>`48.98`|`66.33`<br>`66.19`<br>`20.6`|
