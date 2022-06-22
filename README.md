# LPT
Official implementation of the CVPR2022 paper "Learning of Global Objective for Network Flow in Multi-Object Tracking"
Code coming soon!

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


### Training
Run run_qp.ipynb.

### Inference
Run run_test.ipynb, you need to adjust the MOT17/20 dataset path accordingly.

- **To Do List:**
Upload data and so on.

If you have any questions using this code, please open an issue. I'll respond ASAP.
