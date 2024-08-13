if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from typing import List
from diffusion_policy.dataset.multitask_dataset import MultiDataLoader
from itertools import zip_longest
import psutil
import mimicgen
import time
import gc 
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if_train = False
        if_eval = True


        # lastest_ckpt_path = pathlib.Path("/home/yixiao/yixiao/sdp/SDP/outputs/2024-08-12/07-36-17/checkpoints/50.ckpt")
        # lastest_ckpt_path = pathlib.Path("/home/yixiao/yixiao/sdp/SDP/outputs/2024-08-12/07-36-17/checkpoints/100.ckpt")


        # lastest_ckpt_path = pathlib.Path("/home/yixiao/yixiao/sdp/SDP/outputs/2024-08-12/15-03-03/checkpoints/150.ckpt")
        # lastest_ckpt_path = pathlib.Path("/home/yixiao/yixiao/sdp/SDP/outputs/2024-08-12/15-03-03/checkpoints/200.ckpt")
        # lastest_ckpt_path = pathlib.Path("/home/yixiao/yixiao/sdp/SDP/outputs/2024-08-12/15-03-03/checkpoints/250.ckpt")
        
        lastest_ckpt_path = pathlib.Path("/home/yixiao/yixiao/sdp/SDP/outputs/2024-08-12/15-03-03/checkpoints/275.ckpt")

        # lastest_ckpt_path = pathlib.Path("/home/yixiao/yixiao/sdp/SDP/outputs/2024-08-13/09-14-31/checkpoints/300.ckpt")

        
        # lastest_ckpt_path = pathlib.Path("/home/yixiao/yixiao/sdp/SDP/outputs/2024-08-13/07-40-25/checkpoints/latest.ckpt")

        self.load_checkpoint(path=lastest_ckpt_path)
        # resume training
        # if cfg.training.resume:   
        #     lastest_ckpt_path = pathlib.Path("/home/yixiao/projects/sdp/SDP/outputs/2024-08-06/20-21-25/checkpoints/latest.ckpt")
        #     if lastest_ckpt_path.is_file():
        #         print(f"Resuming from checkpoint {lastest_ckpt_path}")
        #         self.load_checkpoint(path=lastest_ckpt_path)

        # mem=psutil.virtual_memory()
        # print('before current available memory is' +' : '+ str(round(mem.used/1024**2)) +' MIB')
        # configure dataset
        if if_train:
            datasets: List[BaseImageDataset] = []
            for i in range(cfg.task_num):
                datasets.append(hydra.utils.instantiate(cfg[f'task{i}'].dataset))
            
            assert isinstance(datasets[0], BaseImageDataset)
            train_dataloaders = []
            
            normalizers=[]
            for dataset in datasets:
                train_dataloaders.append(DataLoader(dataset, **cfg.dataloader))
                normalizers.append(dataset.get_normalizer())
                
            max_train_dataloader_len = max([len(train_dataloader) for train_dataloader in train_dataloaders])
            for train_dataloader in train_dataloaders:
                print("Length of train_dataloader: ", len(train_dataloader))
            multi_traindataloader=MultiDataLoader(train_dataloaders)
            multi_traindataloader.get_memory_usage()
            
            val_datasets=[]
            for dataset in datasets:
                val_datasets.append(dataset.get_validation_dataset())
        
            val_dataloaders = []
            for val_dataset in val_datasets:
                val_dataloaders.append(DataLoader(val_dataset, **cfg.val_dataloader))


        # for i in range(len(normalizers)):
        #     torch.save(normalizers[i].state_dict(),'norm'+str(i)+'.ckpt')
            
        # exit()

        if if_eval:
            normalizers = []
            for i in range(8):
                normalizer = LinearNormalizer()
                normalizer.load_state_dict(torch.load('norm'+str(i)+'.ckpt'))
                normalizers.append(normalizer)


        self.model.set_normalizer(normalizers)
        
        
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizers)

        # configure lr scheduler
        if if_train:
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=(
                    max_train_dataloader_len * cfg.training.num_epochs) \
                        // cfg.training.gradient_accumulate_every,
                # pytorch assumes stepping LRScheduler every epoch
                # however huggingface diffusers steps it every batch
                last_epoch=self.global_step-1
            )
        
        
        

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # # configure env
        # env_runners = []
        # # env_runner3: BaseImageRunner
        # for i in range(cfg.task_num):
        #     env_runners.append(hydra.utils.instantiate(cfg[f'task{i}'].env_runner, output_dir=self.output_dir))
        #     assert isinstance(env_runners[i], BaseImageRunner)


        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
        
        # del datasets
        # del train_dataloaders
        # del multi_traindataloader
        # del val_datasets
        # del val_dataloaders
        
        
        # gc.collect()
        # print('sleep 2s')
        # time.sleep(2)
        

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        for normalizer in self.model.normalizers:
            normalizer.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
            for normalizer in self.ema_model.normalizers:
                normalizer.to(device)
        optimizer_to(self.optimizer, device)
        
        # save batch for sampling
        train_sampling_batchs = []
        for i in range(cfg.task_num):
            train_sampling_batchs.append(None)
        # train_sampling_batch3 = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()

                
                if if_train:
                    with tqdm.tqdm(multi_traindataloader, desc=f"Training epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx,batch in enumerate(tepoch):

                            assigned_task_id = batch_idx%cfg.task_num
                            

                            # load the next batch of the dataloader, 'DataLoader' object is not an iterator
                            assert assigned_task_id == multi_traindataloader.loader_idx
                            if batch is None:
                                continue
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            task_id = torch.tensor([assigned_task_id], dtype=torch.int64).to(device)
                            if train_sampling_batchs[assigned_task_id] is None:
                                print("Assigning train_sampling_batch with task_id: ", assigned_task_id)
                                train_sampling_batchs[assigned_task_id] = batch
                    

                            # compute loss
                            raw_loss = self.model.compute_loss(batch,task_id)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                            # step optimizer
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()
                            
                            # update ema
                            if cfg.training.use_ema:
                                ema.step(self.model)

                            # logging
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }

                            is_last_batch = (batch_idx == (max_train_dataloader_len-1))
                            if not is_last_batch:
                                # log of last step is combined with validation and rollout
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                                self.global_step += 1

                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break

                    for i, train_sampling_batch in enumerate(train_sampling_batchs):
                        if train_sampling_batch is None:
                            raise ValueError(f"train_sampling_batch {i} is None")
                    # at the end of each epoch
                    # replace train_loss with epoch average
                    train_loss = np.mean(train_losses)
                    step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                runner_logs = []
                # if ((self.epoch+1) % cfg.training.rollout_every) == 0:
                    
                #     for i in range(cfg.task_num):
                #         env_runner = hydra.utils.instantiate(cfg[f'task{i}'].env_runner, output_dir=self.output_dir)
                #         runner_log = env_runner.run(policy,task_id=torch.tensor([i], dtype=torch.int64).to(device))
                #         runner_log = {key + f'_{i}': value for key, value in runner_log.items()}
                #         runner_logs.append(runner_log)
                #     for runner_log in runner_logs:
                #         step_log.update(runner_log)
                

                if if_eval:
                    scores = []

                    for i in range(cfg.task_num):
                        env_runner = hydra.utils.instantiate(cfg[f'task{i}'].env_runner, output_dir=self.output_dir)
                        runner_log = env_runner.run(policy,task_id=torch.tensor([i], dtype=torch.int64).to(device))
                        runner_log = {key + f'_{i}': value for key, value in runner_log.items()}
                        print(i,runner_log)
                        runner_logs.append(runner_log)
                        scores.append(runner_log['test/mean_score_'+str(i)])
                    for runner_log in runner_logs:
                        step_log.update(runner_log)

                    print(scores)
                    exit()
                    
                        
                    
                env_runner = None
                # run validation
                # if (self.epoch % cfg.training.val_every) == 0:
                #     with torch.no_grad():
                #         val_losses_list = []
                #         for i in range(cfg.task_num):
                #             val_losses_list.append([])
                #         zip_val_dataloaders = zip_longest(*val_dataloaders)
                #         # val_losses3 = list()
                #         with tqdm.tqdm(zip_val_dataloaders, desc=f"Validation epoch {self.epoch}", 
                #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batches in enumerate(tepoch):
                #                 for i, batch in enumerate(batches):
                #                     if batch is None:
                #                         continue
                #                     batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                #                     loss = self.model.compute_loss(batch,task_id=torch.tensor([i], dtype=torch.int64).to(device))
                #                     val_losses_list[i].append(loss)
                #                     if (cfg.training.max_val_steps is not None) \
                #                         and batch_idx >= (cfg.training.max_val_steps-1):
                #                         break
                #         if len(val_losses_list[0]) > 0:
                #             for i, val_losses in enumerate(val_losses_list):
                #                 val_loss = torch.mean(torch.tensor(val_losses)).item()
                #                 step_log[f'val_loss_{i}'] = val_loss
                #             # step_log['val_loss3'] = val_loss3
                # # run diffusion sampling on a training batch
                # if (self.epoch % cfg.training.sample_every) == 0:
                #     with torch.no_grad():
                #         for i, train_sampling_batch in enumerate(train_sampling_batchs):
                #             assert train_sampling_batch is not None
                #             batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                #             obs_dict = batch['obs']
                #             gt_action = batch['action']
                #             result = policy.predict_action(obs_dict,task_id=torch.tensor([i], dtype=torch.int64).to(device))
                #             pred_action = result['action_pred'] 
                #             mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                #             step_log[f'train_action_mse_error_{i}'] = mse.item()
                #             del batch
                #             del obs_dict
                #             del gt_action
                #             del result
                #             del pred_action
                #             del mse
                
                # checkpoint
                self.save_checkpoint()
                
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    self.save_checkpoint(tag = str(self.epoch))
                    
                    # # checkpointing
                    # if cfg.checkpoint.save_last_ckpt:
                    #     self.save_checkpoint()
                    # if cfg.checkpoint.save_last_snapshot:
                    #     self.save_snapshot()

                    # # sanitize metric names
                    # metric_dict = dict()
                    # for key, value in step_log.items():
                    #     new_key = key.replace('/', '_')
                    #     metric_dict[new_key] = value
                    # sum=0
                    # for key in metric_dict.keys():
                    #     # if start with cfg.checkpoint.topk.monitor_key, then sum up
                    #     if key.startswith(cfg.checkpoint.topk.monitor_key):
                    #         sum+=metric_dict[key]
                    # metric_dict[cfg.checkpoint.topk.monitor_key] = sum
                    
                    # # We can't copy the last checkpoint here
                    # # since save_checkpoint uses threads.
                    # # therefore at this point the file might have been empty!
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # if topk_ckpt_path is not None:
                    #     self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                multi_traindataloader.reset()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
