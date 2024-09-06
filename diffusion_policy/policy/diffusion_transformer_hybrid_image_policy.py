from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from patch_moe.encoder import MoEEncoder
from diffusion_policy.model.diffusion.encoder import StateEncoder

from patch_moe.resnet import ResNet, PatchMoeResNet
import robomimic
class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_tasks,
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.my_obs = True

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        
        
        # k = [4,4,2,2]
        # exp = [8,8,4,4]
        # ### [16,16,8,8]
        # patch_size = [2,2,2,2]
        # n_blocks_list = [2,2,2,2]
        

        # obs_encoder.obs_nets.agentview_image.backbone = PatchMoeResNet(k = k, 
        #                                                                exp = exp, 
        #                                                                patch_size=patch_size ,
        #                                                                n_blocks_list=n_blocks_list)
        
        # obs_encoder.obs_nets.robot0_eye_in_hand_image.backbone = PatchMoeResNet(k = k, 
        #                                                                         exp = exp, 
        #                                                                         patch_size=patch_size ,
        #                                                                         n_blocks_list=n_blocks_list)
        
        
        def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Get the number of parameters
        num_params = count_parameters(obs_encoder)
        print(f'The ob1 has {num_params/1000000} trainable parameters')
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            
        # self.my_obs_encoder = StateEncoder()
        self.my_obs_encoder = None
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, robomimic.models.obs_core.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_tasks=n_tasks,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizers = [] #LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, task_id=None,generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond,task_id)
            model_output = model_output[0]
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor],task_id) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizers[task_id].normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))

            if self.my_obs_encoder is None:
                nobs_features = self.obs_encoder(this_nobs)
            else:
                out_dic_rand = {}
                for key in self.obs_encoder.obs_randomizers.keys():
                    x = this_nobs[key]
                    if self.obs_encoder.obs_randomizers[key] is not None:
                        x = self.obs_encoder.obs_randomizers[key].forward_in(x)
                    out_dic_rand[key] = x
                nobs_features, obs_mi_loss = self.my_obs_encoder(out_dic_rand, task_id)
            
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            if self.my_obs_encoder is None:
                nobs_features = self.obs_encoder(this_nobs)
            else:
                out_dic_rand = {}
                for key in self.obs_encoder.obs_randomizers.keys():
                    x = this_nobs[key]
                    if self.obs_encoder.obs_randomizers[key] is not None:
                        x = self.obs_encoder.obs_randomizers[key].forward_in(x)
                    out_dic_rand[key] = x
                nobs_features, obs_mi_loss = self.my_obs_encoder(out_dic_rand, task_id)

            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            task_id=task_id,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizers[task_id]['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizers):
        for normalizer in normalizers:
            self.normalizers.append(LinearNormalizer())
            self.normalizers[-1].load_state_dict(normalizer.state_dict())
            
    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch,task_id):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizers[task_id].normalize(batch['obs'])
        nactions = self.normalizers[task_id]['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        
        # agentview_image torch.Size([128, 3, 84, 84]) after [3,80,80]
        # robot0_eye_in_hand_image torch.Size([128, 3, 84, 84])
        # robot0_eef_pos torch.Size([128, 3])
        # robot0_eef_quat torch.Size([128, 4])
        # robot0_gripper_qpos torch.Size([128, 2])
        
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            
            
            
            if self.my_obs_encoder is None:
                nobs_features = self.obs_encoder(this_nobs)
            else:
                out_dic_rand = {}
                for key in self.obs_encoder.obs_randomizers.keys():
                    x = this_nobs[key]
                    if self.obs_encoder.obs_randomizers[key] is not None:
                        x = self.obs_encoder.obs_randomizers[key].forward_in(x)
                    out_dic_rand[key] = x
                nobs_features, obs_mi_loss = self.my_obs_encoder(out_dic_rand, task_id)
            
            # def count_parameters(model):
            #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

            # # Get the number of parameters
            # num_params = count_parameters(self.my_obs_encoder)
            # print(f'The encoder has {num_params/1000000} trainable parameters')
            
            # print(cond.size())
            # print(nobs_features.size())
                
            # exit()
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))

            if self.my_obs_encoder is None:
                nobs_features = self.obs_encoder(this_nobs)
            else:
                out_dic_rand = {}
                for key in self.obs_encoder.obs_randomizers.keys():
                    x = this_nobs[key]
                    if self.obs_encoder.obs_randomizers[key] is not None:
                        x = self.obs_encoder.obs_randomizers[key].forward_in(x)
                    out_dic_rand[key] = x
                nobs_features, obs_mi_loss = self.my_obs_encoder(out_dic_rand, task_id)

            
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred,aux_loss,probs = self.model(noisy_trajectory, timesteps, cond,task_id)

        if self.my_obs_encoder is not None:
            aux_loss += obs_mi_loss

        # Convert the tensor to a NumPy array
        probs_np = probs.detach().cpu().numpy()

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss+aux_loss
