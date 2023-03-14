# LPT

![LPT](figures/main_idea.png)

Official implementation of the CVPR2022 paper "Learning of Global Objective for Network Flow in Multi-Object Tracking"

## Setup

```shell script

# Install cvxpy and qpth
pip install cvxpy
pip install qpth

# Install pytorch_geometric(Version < 2.0.0) and gurobipy, please check if gurobipy is correctly installed
import gurobipy as gp
gurobi_solver = gp.Model()

# Install torchreid from: https://github.com/KaiyangZhou/deep-person-reid, and put it inside ./lib folder.

```

## Data
Download pre-processed detections&appearance features (~1.4GB): [\[Google Drive\]](https://drive.google.com/file/d/1aJBJcBcD9YNvFvLVb_ECP2W8mYBzPpoI/view?usp=sharing)
We aslo provide the tracking results on MOT17/20 test set in txt format, which can replicate the numbers reported in the paper.

## Training
Execute run_train.ipynb. Monitor all relevant metrics, using our provided data, the loss converges at around 8 epochs. Training with more data might be helpful, but not tested.

## Inference
Execute run_test.ipynb, you need to adjust the MOT17/20 data path accordingly. Note that for one specific MOT20 sequence, running time could be long. 

If you have any questions using this code, please open an issue. I'll respond ASAP.

## Citing
If you find this code useful in your research, please consider citing:
```bibtex
@inproceedings{li2022learning,
  title={Learning of Global Objective for Network Flow in Multi-Object Tracking},
  author={Li, Shuai and Kong, Yu and Rezatofighi, Hamid},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8855--8865},
  year={2022}
}
