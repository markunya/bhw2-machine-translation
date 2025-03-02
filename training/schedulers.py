from typing import Literal
from utils.class_registry import ClassRegistry
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR

schedulers_registry = ClassRegistry()

WarmUpCurve = Literal['linear', 'convex', 'concave']

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
                    return current_step / max(1, warmup_steps)
                elif warmup_curve == 'convex':
                    return (current_step / max(1, warmup_steps))**2
                elif warmup_curve == 'concave':
                    return (current_step / max(1, warmup_steps))**0.5
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
                warmup_steps=0, warmup_curve='linear', **kwargs):
        base_scheduler = base_scheduler_type(optimizer, **kwargs)

        if warmup_steps > 0:
            self.scheduler = WarmUpScheduler(optimizer, warmup_steps, warmup_curve, base_scheduler)
        else:
            self.scheduler = base_scheduler

    def step(self, epoch=None):
        self.scheduler.step(epoch)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
@schedulers_registry.add_to_registry(name='multi_step')
class MultiStepScheduler(BaseScheduler):
    def __init__(
            self,
            optimizer,
            warmup_steps=0,
            warmup_curve: WarmUpCurve='linear',
            **kwargs
        ):
        super().__init__(
            optimizer,
            MultiStepLR,
            warmup_steps,
            warmup_curve,
            **kwargs
        )

@schedulers_registry.add_to_registry(name='exponential')
class ExponentialScheduler(BaseScheduler):
    def __init__(
            self,
            optimizer,
            warmup_steps=0,
            warmup_curve: WarmUpCurve='linear',
            **kwargs
        ):
        super().__init__(
            optimizer,
            ExponentialLR,
            warmup_steps,
            warmup_curve,
            **kwargs
        )
