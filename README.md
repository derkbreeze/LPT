# LPT

![LPT](figures/main_idea.png)

Official implementation of the CVPR2022 paper "Learning of Global Objective for Network Flow in Multi-Object Tracking"

### In order to run the code successfully
- **Gurobi Python:**
Please make sure to install Gurobi Python interface first. 
To check whether the install is successful or not, Do following python code

```shell script
import gurobipy as gp
gurobi_solver = gp.Model()
```

If the installation is successful, there will be no warnings/errors.

- **ReID:**
Please install torchreid, it can be downloaded from: [\[torchreid\]](https://github.com/KaiyangZhou/deep-person-reid). and put it inside ./lib folder.

### Data
Download pre-processed detections&appearance(aka ReID) features (~2.1GB): [\[Google Drive\]](https://drive.google.com/drive/folders/1GdpxkEevzdDC04k5AATgrxhrwp3MIS-A?usp=sharing)

### Training
Run run_qp.ipynb.

### Inference
Run run_test.ipynb, you need to adjust the MOT17/20 dataset path accordingly.

If you have any questions using this code, please open an issue. I'll respond ASAP.

## Citing Unicorn
If you find this code useful in your research, please consider citing:
```bibtex
@inproceedings{Li2022Learning,
  title={Learning of Global Objective for Network Flow in Multi-Object Tracking},
  author={Li, Shuai and Kong, Yu and Rezatofighi, Hamid},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8855--8865},
  year={2022}
}
