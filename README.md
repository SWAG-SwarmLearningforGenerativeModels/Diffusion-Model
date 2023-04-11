# Diffusion Models

A diffusion model repo. Images can be trained `conditionally` or `unconditionally`. 

## 1. Implementation

### 1 Training

#### 1.1 Unconditional Training (Default)

1. Navigate to the config folder, create a config.yml file and specify model and data parameters accordingly
2. Set yaml file and project name in main.py for loggings
3. In dataloaders.py, create a dataloader for your images within the ```get_data``` function 
4. Initialize script with ```python main.py --config config_file.yml ---doc project_name```
5. At different snapshot_frequencies, save model (snapshot_freq parameter is set in ```config_file.yml file```)

#### 1.2 Conditional Training

1. The --conditonal attribute is set to False by default.
2. Initialize script with ```python main.py --config config_file.yml ---doc project_name --conditional True```

#### 1.3 Resume Training
1. Resume tarining using the script ```python main.py --config config_file.yml ---doc project_name --resume_training True```


## 2. Sampling

1. To generate samples conditionally and unconditionally, use script ``python main.py --config config_file.yml ---doc project_name --sample True```
2. Images are stored in ```.../exp/logs/rsna/unconditional/samples``` path

### 2.1 X-Ray (Unconditional)
Generated chest x-ray images (224 x 224 resolutions)

![xray](https://user-images.githubusercontent.com/77448406/212916216-a8715296-e1f5-4899-9766-0c513570d7bb.png)

### 2.2 Adbomen CT (Unconditional)
Generated CT images of the abdomen (224 x 224 resolutions)

![4](https://user-images.githubusercontent.com/77448406/231154615-db03e021-883f-4339-8f76-a96e62461295.jpg)


Reference: [labml](https://github.com/labmlai/annotated_deep_learning_paper_implementations) and [dome272](https://github.com/dome272)