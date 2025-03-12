import math
from typing import Literal
from utils.class_registry import ClassRegistry
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR

schedulers_registry = ClassRegistry()

WarmUpCurve = Literal['linear', 'convex', 'concave']
ReduceLrTime = Literal['epoch', 'step', 'period']

class WarmUpScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, warmup_curve, base_scheduler, **kwargs):
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.finished_warmup = False
        self.current_step = 1

        def lr_lambda(current_step):
            current_step += 1
            if current_step <= warmup_steps:
                if warmup_curve == 'linear':
                    return current_step / max(1, warmup_steps + 1)
                elif warmup_curve == 'convex':
                    return (current_step / max(1, warmup_steps + 1))**2
                elif warmup_curve == 'concave':
                    return (current_step / max(1, warmup_steps + 1))**0.5
            else:
                self.finished_warmup = True
                return 1.0

        super().__init__(optimizer, lr_lambda, **kwargs)

    def step(self, epoch=None):
        if self.finished_warmup:
            self.base_scheduler.step(epoch)
        else:
            self.current_step += 1
            super().step(epoch)

    def get_last_lr(self):
        if self.finished_warmup:
            return self.base_scheduler.get_last_lr()
        return super().get_last_lr()
    
class BaseScheduler:
    def __init__(self, optimizer, base_scheduler_type,
                reduce_time: ReduceLrTime, step_period=None,
                warmup_steps=0, warmup_curve='linear', **kwargs):
        base_scheduler = base_scheduler_type(optimizer, **kwargs)
        self.reduce_time = reduce_time
        if reduce_time == 'period':
            self.period = step_period

        if warmup_steps > 0:
            self.scheduler = WarmUpScheduler(optimizer, warmup_steps, warmup_curve, base_scheduler)
        else:
            self.scheduler = base_scheduler

    def step(self, epoch=None):
        self.scheduler.step(epoch)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, *args, **kwargs):
        return self.scheduler.load_state_dict(*args, **kwargs)
    
@schedulers_registry.add_to_registry(name='multi_step')
class MultiStepScheduler(BaseScheduler):
    def __init__(
            self,
            optimizer,
            reduce_time: ReduceLrTime,
            step_period=None,
            warmup_steps=0,
            warmup_curve: WarmUpCurve='linear',
            **kwargs
        ):
        super().__init__(
            optimizer,
            MultiStepLR,
            reduce_time,
            step_period,
            warmup_steps,
            warmup_curve,
            **kwargs
        )

@schedulers_registry.add_to_registry(name='exponential')
class ExponentialScheduler(BaseScheduler):
    def __init__(
            self,
            optimizer,
            reduce_time: ReduceLrTime,
            step_period=None,
            warmup_steps=0,
            warmup_curve: WarmUpCurve='linear',
            **kwargs
        ):
        super().__init__(
            optimizer,
            ExponentialLR,
            reduce_time,
            step_period,
            warmup_steps,
            warmup_curve,
            **kwargs
        )

@schedulers_registry.add_to_registry(name='cosine')
class CosineScheduler(BaseScheduler):
    def __init__(
            self,
            optimizer,
            reduce_time: ReduceLrTime,
            step_period=None,
            warmup_steps=0,
            warmup_curve: WarmUpCurve = 'linear',
            total_steps=1000,
            eta_min=0.0,
            **kwargs
        ):
        def lr_lambda(current_step: int):
            if current_step >= total_steps:
                return eta_min
            progress = current_step / total_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return eta_min + (1 - eta_min) * cosine_decay

        super().__init__(
            optimizer=optimizer,
            base_scheduler_type=LambdaLR,
            reduce_time=reduce_time,
            step_period=step_period,
            warmup_steps=warmup_steps,
            warmup_curve=warmup_curve,
            lr_lambda=lr_lambda,
            **kwargs
        )
