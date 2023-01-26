# Diffusion Models

A diffusion model repo. Images can be trained `conditionally` or `unconditionally`. 

## Implementation

### Unconditional Training

1. Navigate to the config folder, create a config.yml file and specify model and data parameters accordingly
2. Set yaml file and project name in main.py for loggings
3. In dataloaders.py, create a dataloader for your images within the ```get_data``` function 
4. Initialize script with ```python main.py --config 'config_file_name.yml' ---doc my_project_name```

## Sampling

### X-Ray (Unconditional)
Generated chest x-ray images (224 x 224 resolutions)

![xray](https://user-images.githubusercontent.com/77448406/212916216-a8715296-e1f5-4899-9766-0c513570d7bb.png)


Reference: [labml](https://github.com/labmlai/annotated_deep_learning_paper_implementations) and [dome272](https://github.com/dome272)