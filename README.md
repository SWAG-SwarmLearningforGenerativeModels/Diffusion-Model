# Diffusion Models

A diffusion model repo. Images can be trained `conditionally` or `unconditionally`. 

## 1. This repo has the folder structure

<pre>
```
project_root/
    ├── configs/
    │     ├── model.yml
    │     ├── abdomen.yml
    │     └── xray.yml
    ├── data/
    │     ├── abdomen_ct/
    │     └── xray/
    ├── runner/
    │     ├── ddpm_conditional.py
    │     └── conditional.py
    ├── unet/
    │     ├── __init__.py
    │     └── modules.py
    ├── dataloaders.py
    ├── main.py
    ├── README.md
    ├── requirements.txt
    └── utils.py
```
</pre>

### 2. Training

#### 2.1 - Unconditional Training (Default)

1. Navigate to the configs folder, create a config_file.yml file and set training, sampling, data and optimizer parameters accordingly.
2. In dataloaders.py, create a dataloader for your images within the ```get_data``` function. 
3. The model.yml file in the configs folder is strictly for the diffusion model parameter which can also be adjusted.
4. In main.py file, you can parse default command-line arguments for the project and loggings. 
5. Initialize script with ```python main.py --config config_file.yml ---doc project_name```.
6. Project info is logged in ```.../exp/logs/project_name/unconditional/...``` path.
7. Save model (snapshot_freq parameter is set under ```training``` attribute in the ```config_file.yml file```).

#### 2.2 - Conditional Training

1. The --conditonal attribute is set to False by default.
2. Set ```num_classes``` attribute correctly in ```data``` attribute in the ```config_file.yml file```.
3. Project info is logged in ```.../exp/logs/project_name/conditional/..``` path.
4. Initialize script with ```python main.py --config config_file.yml ---doc project_name --conditional True```.

#### 2.3 - Resume Training
1. Resume training ```python main.py --config config_file.yml ---doc project_name --resume_training True```.


## 3. Sampling

1. To generate samples  ```python main.py --config config_file.yml ---doc project_name --sample True```.
2. Sampled images are stored in ```.../exp/logs/project_name/(un)conditional/samples``` path.

### 3.1 - X-Ray (Unconditional)
Generated chest x-ray images (224 x 224 resolutions)

![xray](https://user-images.githubusercontent.com/77448406/212916216-a8715296-e1f5-4899-9766-0c513570d7bb.png)

### 3.2 - Adbomen CT (Unconditional)
Generated CT images of the abdomen (224 x 224 resolutions)

![4](https://user-images.githubusercontent.com/77448406/231154615-db03e021-883f-4339-8f76-a96e62461295.jpg)


Reference: [labml](https://github.com/labmlai/annotated_deep_learning_paper_implementations) and [dome272](https://github.com/dome272)