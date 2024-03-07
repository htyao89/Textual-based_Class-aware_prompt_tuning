## TCP: Textual-based Class-aware Prompt tuning for Visual-Language Model[CVPR24]

> [**TCP: Textual-based Class-aware Prompt tuning for Visual-Language Model**](https://arxiv.org/abs/2311.18231)<br>
> Hantao Yao, Rui Zhang, Changsheng Xu

## How to Install
This code is built on top of the toolbox [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). You can prepare the environment as follows:

```
# Create a conda environment
conda create -n dassl python=3.7

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
# Please make sure you have installed the gpu version due to the speed.
# For example:
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

After that, run `pip install -r requirements.txt` under `Textual-based_Class-aware_prompt_tuning/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## [Importantly]Adjust `EPS` in Adam optimzier
Since using the standard AdaW on the fp16 data will produce NaN loss, we thus set the EPS in AdaW as 1e-3. The discussion can also be see https://discuss.pytorch.org/t/adam-half-precision-nans/1765.

Line 80: ./Dassl.pytorch/dassl/optim/optimizer.py

```
if optim == "adam":
    optimizer = torch.optim.Adam(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2),
        eps=1e-3,
    )
```




## Generalization From Base to New Classes

You will need `base2new_train_main.sh`. The scripts with the prefix `base2new_train` train a model on base classes while the ones with the prefix `base2new_test` evaluate the trained model on new classes. Both kinds of scripts have only one input argument, i.e., `DATASET`. `DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `CoOp/configs/datasets/`.

Below we provide an example on how to evaluate the model on ImageNet.

```bash
bash base2new_train.sh
```

When the evaluation is done, you can use `parse_test_res.py` to automatically calculate the average results. For instance, after you finish the evaluation using the aforementioned commands, you would get


Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/stanford_cars/shots_16/CoCoOp/rn50_ep100
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/stanford_cars/shots_16/CoCoOp/rn50_ep100 --test-log
```

## Citation
If you use our work, please consider citing:
```bibtex
@inproceedings{TCP24,
    title={TCP: Textual-based Class-aware Prompt tuning for Visual-Language Model},
    author={Hantao Yao, Rui Zhang, Changsheng Xu},
    booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```

