python train.py --config-dir=. --config-name=image_multi_task_diffusion_policy_transformer.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


python train.py --config-dir=. --config-name=image_multi_task_diffusion_policy_transformer.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='/mnt/hdd1/spdiff/data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'