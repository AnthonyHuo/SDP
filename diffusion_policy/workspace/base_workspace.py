from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, filter=False, **kwargs):
        if exclude_keys is None:
            if filter:
                exclude_keys = tuple(['optimizer','cfg'])
            else:
                exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()
        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                if filter:
                    print(f"Filtering {key}")
                    if key=='model' or key=='ema_model':
                        model_value_copy = copy.deepcopy(payload['state_dicts']['ema_model']) # resume the ema value
                        
                        for k in value.keys():
                            if k.endswith('experts.old_weight'):
                                # combine with new weight
                                model_value_copy[k]= torch.cat(
                                    [payload['state_dicts']['ema_model'][k],
                                    payload['state_dicts']['ema_model'][k.replace('old_weight','new_weight')]],
                                    dim=0)
                                del model_value_copy[k.replace('old_weight','new_weight')]

                            if k.endswith('experts.old_bias'):
                                # combine with new bias
                                model_value_copy[k]= torch.cat(
                                    [payload['state_dicts']['ema_model'][k],
                                    payload['state_dicts']['ema_model'][k.replace('old_bias','new_bias')]],
                                    dim=0)
                                del model_value_copy[k.replace('old_bias','new_bias')]
                            
                            # if key end with experts.weight, rename it to experts.old_weight
                            if k.endswith('experts.weight'):
                                model_value_copy[k.replace('experts.weight', 'experts.old_weight')] = payload['state_dicts']['ema_model'][k]
                                del model_value_copy[k]
                            if k.endswith('experts.bias'):
                                model_value_copy[k.replace('experts.bias', 'experts.old_bias')] = payload['state_dicts']['ema_model'][k]
                                del model_value_copy[k]
                            
                            if k.endswith('PE') or k.endswith('PTE'):
                                # align shape
                                target_shape=self.__dict__[key].state_dict()[k].shape
                                new_value=torch.zeros(target_shape)
                                old_value=payload['state_dicts']['ema_model'][k]
                                if k.endswith('PE'):
                                    new_value[:old_value.shape[0]]=old_value
                                else:
                                    new_value[:old_value.shape[0],:old_value.shape[1]]=old_value
                                model_value_copy[k]=new_value
                                # del model_value_copy[k]
                        
                        self.__dict__[key].load_state_dict(model_value_copy, strict=False, **kwargs)
                    else:
                        try:
                            self.__dict__[key].load_state_dict(value, **kwargs)
                        except:
                            print(f"Error in loading {key}")
                            raise Exception
                else:
                    print(f"Not filtering {key}")
                    if key=='model' or key=='ema_model':
                        model_value_copy = copy.deepcopy(value)
                        for k in value.keys():
                            if k.endswith('task_moe_layer.experts.weight') \
                            or k.endswith('task_moe_layer.experts.bias')\
                            or k.endswith('task_moe_layer.output_experts.weight')\
                            or k.endswith('task_moe_layer.output_experts.bias')\
                            :
                                del model_value_copy[k]
                        self.__dict__[key].load_state_dict(model_value_copy,strict=True, **kwargs)
                    else:
                        self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                if filter:
                    continue
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            filter=False,
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys, filter=filter)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
