# Sparse Diffusion Policy

### Dataset Download

Using Direct Download Links

You can download the datasets manually through Google Drive. The folders each correspond to the dataset types described in [Dataset Types](#dataset-types).

**Google Drive folder with all mimicgen datasets:** [link](https://drive.google.com/drive/folders/14e9kkHGfApuQ709LBEbXrXVI1Lp5Ax7p?usp=drive_link)

Then, you should download the dataset with core folder in the path robomimic/core.

### 🛠️ Installation
#### 🖥️ Simulation
To reproduce our simulation benchmark results, install our conda environment on a Linux machine with Nvidia GPU. On Ubuntu 20.04 you need to install the following apt packages for mujoco:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f conda_environment.yaml
```

but you can use conda as well: 
```console
$ conda env create -f conda_environment.yaml
```
### Installation

Then we install the packages for mimicgen:

```sh
conda activate sdp
```

You can install most of the dependencies by cloning the repository and then installing from source:

```sh
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone https://github.com/NVlabs/mimicgen_environments.git
cd mimicgen_environments
pip install -e .
```

There are some additional dependencies that we list below. These are installed from source:

- [robosuite](https://robosuite.ai/)
    - **Installation**
      ```sh
      cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
      git clone https://github.com/ARISE-Initiative/robosuite.git
      cd robosuite
      git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa
      pip install -e .
      ```
    - **Note**: the git checkout command corresponds to the commit we used for testing our policy learning results. In general the `master` branch (`v1.4+`) should be fine.
    - For more detailed instructions, see [here](https://robosuite.ai/docs/installation.html)
- [robomimic](https://robomimic.github.io/)
    - **Installation**
      ```sh
      cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
      git clone https://github.com/ARISE-Initiative/robomimic.git
      cd robomimic
      git checkout ab6c3dcb8506f7f06b43b41365e5b3288c858520
      pip install -e .
      ```
    - **Note**: the git checkout command corresponds to the commit we used for testing our policy learning results. In general the `master` branch (`v0.3+`) should be fine.
    - For more detailed instructions, see [here](https://robomimic.github.io/docs/introduction/installation.html)
- [robosuite_task_zoo](https://github.com/ARISE-Initiative/robosuite-task-zoo)
    - **Note**: This is optional and only needed for the Kitchen and Hammer Cleanup environments / datasets.
    - **Installation**
      ```sh
      cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
      git clone https://github.com/ARISE-Initiative/robosuite-task-zoo
      cd robosuite-task-zoo
      git checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
      pip install -e .
      ```

Lastly, **please downgrade MuJoCo to 2.3.2**:
```sh
pip install mujoco==2.3.2
```

**Note**: This MuJoCo version (`2.3.2`) is important -- in our testing, we found that other versions of MuJoCo could be problematic, especially for the Sawyer arm datasets (e.g. `2.3.5` causes problems with rendering and `2.3.7` changes the dynamics of the robot arm significantly from the collected datasets).

The `conda_environment_macos.yaml` file is only for development on MacOS and does not have full support for benchmarks.

### Training
```console
$ python train.py
```
The results in our paper is evaluated every 50 epochs, after 100 epochs, you can get a result similar in our paper.

### Training Checkpoints

Within each experiment directory you may find in outputs folder:
```

├── config.yaml
├── metrics
│   └── logs.json.txt
├── train
│   ├── checkpoints
│   │   ├── epoch=0299-test_mean_score=6.070.ckpt
│   │   └── latest.ckpt
│   └── logs.json.txt

```

### Checkpoints

You can download ours SDP checkpoints manually through Google Drive. 

**Google Drive folder with our checkpoints:** [link](https://drive.google.com/file/d/1So-byi2hNXIrPLsMT1KLaSbJTpRM1pil/view)

You can reload the link if it does not work.

You can save the checkpoints in /path/to/ckpt.

### Evaluation
```console
$ python eval.py --checkpoint /path/to/ckpt
```

Then you can get a similar multi-task results in our paper.
