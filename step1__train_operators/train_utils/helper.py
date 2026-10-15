from    typing      import  Optional
from    datetime    import  timedelta
import  torch


__all__: list[str] = ['sec_to_hms', 'hms_to_sec', 'uniform_grid']


def sec_to_hms(seconds: float) -> str:
    return str(timedelta(seconds=seconds))

def hms_to_sec(hours: int, minutes: int, seconds: int) -> int:
    return int(timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds())

def uniform_grid(
        dimension:  int,
        resolution: int,
        max_value:  float,
        dtype:      Optional[torch.dtype]   = None,
        device:     Optional[torch.device]  = None,
    ) -> torch.Tensor:
    if dtype is None:   dtype   = torch.get_default_dtype()
    if device is None:  device  = torch.get_default_device()
    dx = 2*max_value/resolution
    max_x = max_value-dx/2
    grid_1d = torch.linspace(-max_x, max_x, resolution, dtype=dtype, device=device)
    mesh = torch.meshgrid(*(grid_1d for _ in range(dimension)), indexing='ij')
    return torch.stack(mesh, dim=-1)