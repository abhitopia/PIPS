import torch



class AccelerationConfig:
    """Configuration class for training acceleration and optimization settings."""

    VALID_DEVICES = ['auto', 'cpu', 'cuda', 'mps']
    VALID_PRECISIONS = ['32-true', '32', '16-mixed', 'bf16-mixed', 'bf16-true', 'bf16']
    VALID_MATMUL_PRECISIONS = ['highest', 'high', 'medium']
    
    def __init__(
        self,
        device: str = 'auto',
        precision: str = 'bf16-true',
        matmul_precision: str = 'high',
        compile_model: bool = True,
    ):
        """Initialize acceleration config.
        
        Args:
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            precision: Training precision
            matmul_precision: Matrix multiplication precision
            compile_model: Whether to use torch.compile
            _suppress_logs: Internal parameter to suppress logging during validation
        """
        self._validate_inputs(device, precision, matmul_precision)
        
        # Resolve device
        self.device = self._resolve_device(device)
        
        # Resolve precision based on device
        self.precision = self._resolve_precision(precision)
        
        self.matmul_precision = matmul_precision
        self.compile_model = compile_model
        
        self._log_settings()
    
    def _validate_inputs(self, device: str, precision: str, matmul_precision: str):
        """Validate input parameters."""
        if device not in self.VALID_DEVICES:
            raise ValueError(f"Invalid device: {device}. Must be one of {self.VALID_DEVICES}")
        if precision not in self.VALID_PRECISIONS:
            raise ValueError(f"Invalid precision: {precision}. Must be one of {self.VALID_PRECISIONS}")
        if matmul_precision not in self.VALID_MATMUL_PRECISIONS:
            raise ValueError(f"Invalid matmul_precision: {matmul_precision}. Must be one of {self.VALID_MATMUL_PRECISIONS}")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def _resolve_precision(self, precision: str) -> str:
        """Resolve precision based on device."""
        if self.device == 'cpu':
            if precision != '32-true':
                return '32-true'
        return precision
    
    def _log_settings(self):
        """Log the final acceleration settings."""
        print(f"Auto-selected device: {self.device}")
        print(f"Using precision: {self.precision}")
        if self.compile_model:
            print(f"Model compilation enabled")
    
    def apply_settings(self):
        """Apply the acceleration settings."""
        if self.matmul_precision:
            torch.set_float32_matmul_precision(self.matmul_precision)

    def __str__(self):
        return (
            f"AccelerationConfig("
            f"device='{self.device}', "
            f"precision='{self.precision}', "
            f"matmul_precision='{self.matmul_precision}', "
            f"compile_model={self.compile_model})"
        )
    
    def to_dict(self):
        return {
            'device': self.device,
            'precision': self.precision,
            'matmul_precision': self.matmul_precision,
            'compile_model': self.compile_model
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)
