import numpy as np
from typing import Callable

class Schedule:
    """Generic scheduler for parameter annealing or decay"""
    
    @staticmethod
    def get_schedule(
        initial_value: float, 
        target_value: float, 
        transition_steps: int,
        warmup_steps: int = 0,
        schedule_type: str = 'linear'
    ) -> Callable[[int], float]:
        """
        Returns a schedule function that takes global_step as input and returns the current value.
        
        Args:
            initial_value: Starting value
            target_value: Final value
            transition_steps: Number of steps to transition from initial to target value
            warmup_steps: Number of steps to wait before starting the transition (default: 0)
            schedule_type: Type of schedule ('constant', 'linear', 'exponential', 'cosine', or 'threshold')
        """
        def create_base_schedule() -> Callable[[int], float]:
            """Creates the base schedule without warmup handling"""
            if schedule_type == 'constant':
                assert initial_value == target_value, "constant schedule must have initial and target values equal"
                def base_schedule(step: int) -> float:
                    return initial_value  # Always return initial value
                    
            elif schedule_type == 'linear':
                def base_schedule(step: int) -> float:
                    if step >= transition_steps:
                        return target_value
                    return initial_value + (target_value - initial_value) * (step / transition_steps)
                    
            elif schedule_type == 'exponential':
                decay_rate = -np.log(target_value / initial_value) / transition_steps
                def base_schedule(step: int) -> float:
                    if step >= transition_steps:
                        return target_value
                    return initial_value * np.exp(-decay_rate * step)
                    
            elif schedule_type == 'cosine':
                def base_schedule(step: int) -> float:
                    if step >= transition_steps:
                        return target_value
                    progress = step / transition_steps
                    cosine_term = 0.5 * (1 - np.cos(np.pi * progress))
                    return initial_value + (target_value - initial_value) * cosine_term
                    
            elif schedule_type == 'threshold':
                def base_schedule(step: int) -> float:
                    return target_value if step >= transition_steps else initial_value
            else:
                raise ValueError(f"Unknown schedule type: {schedule_type}")
                
            return base_schedule

        # Create base schedule
        base_schedule = create_base_schedule()
        
        # Apply warmup wrapper if needed
        if warmup_steps > 0:
            def warmed_schedule(step: int) -> float:
                if step < warmup_steps:
                    return initial_value
                return base_schedule(step - warmup_steps)
            return warmed_schedule
        else:
            return base_schedule 