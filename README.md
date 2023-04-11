# Diffusion Models

A diffusion model repo. Images can be trained `conditionally` or `unconditionally`. 

## Implementation

### Training

#### Unconditional Training

1. Navigate to the config folder, create a config.yml file and specify model and data parameters accordingly
2. Set yaml file and project name in main.py for loggings
3. In dataloaders.py, create a dataloader for your images within the ```get_data``` function 
4. Initialize script with ```python main.py --config config_file.yml ---doc project_name```

#### Conditional Training

1. The --conditonal attribute is set to False by default.
2. Initialize script with ```python main.py --config config_file.yml ---doc project_name --conditional True```

## Sampling

Generate samples using script ``python main.py --config config_file.yml ---doc project_name --sample True```

### X-Ray (Unconditional)
Generated chest x-ray images (224 x 224 resolutions)

![xray](https://user-images.githubusercontent.com/77448406/212916216-a8715296-e1f5-4899-9766-0c513570d7bb.png)

### Adbomen CT (Unconditional)
Generated CT images of the abdomen (224 x 224 resolutions)

![4](https://user-images.githubusercontent.com/77448406/231154615-db03e021-883f-4339-8f76-a96e62461295.jpg)


Reference: [labml](https://github.com/labmlai/annotated_deep_learning_paper_implementations) and [dome272](https://github.com/dome272)