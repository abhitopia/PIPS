import numpy as np
from typing import Callable

class Schedule:
    """Generic scheduler for parameter annealing or decay"""
    
    @staticmethod
    def get_schedule(
        initial_value: float, 
        target_value: float, 
        warmup_steps: int,
        schedule_type: str = 'linear'
    ) -> Callable[[int], float]:
        """
        Returns a schedule function that takes global_step as input and returns the current value.
        
        Args:
            initial_value: Starting value
            target_value: Final value
            warmup_steps: Number of steps to reach target value
            schedule_type: Type of schedule ('constant', 'linear', 'exponential', 'cosine_decay', or 'cosine_anneal')
        """
        if schedule_type == 'constant':
            assert initial_value == target_value, "constant schedule must have initial and target values equal"
            def schedule(step: int) -> float:
                return initial_value  # Always return initial value
                
        elif schedule_type == 'linear':
            def schedule(step: int) -> float:
                if step >= warmup_steps:
                    return target_value
                return initial_value + (target_value - initial_value) * (step / warmup_steps)
                
        elif schedule_type == 'exponential':
            decay_rate = -np.log(target_value / initial_value) / warmup_steps
            def schedule(step: int) -> float:
                if step >= warmup_steps:
                    return target_value
                return initial_value * np.exp(-decay_rate * step)
                
        elif schedule_type == 'cosine':
            def schedule(step: int) -> float:
                if step >= warmup_steps:
                    return target_value
                progress = step / warmup_steps
                cosine_term = 0.5 * (1 + np.cos(np.pi * progress))
                return initial_value + (target_value - initial_value) * cosine_term
                
        elif schedule_type == 'threshold':
            def schedule(step: int) -> float:
                return target_value if step >= warmup_steps else initial_value
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        return schedule 