import  warnings
from    typing      import  Sequence, Union, Optional
from    pathlib     import  Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import  torch


base_path = _ROOT


__all__: list[str] = ['load_dataset']


def load_dataset(
        dimension:  int,
        resolution: int,
        alpha:      Sequence[float],
        parts:      Union[int, Sequence[int]],
        n_steps:    Optional[int]   = None,
        coeff:      Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Given the test dimension and resolution with the coefficient of the collision operator, this function identifies the path to the dataset and loads the test dataset.
    
    Arguments:
        `dimension` (`int`): The dimension of the dataset to be loaded.
        `resolution` (`int`): The resolution of the dataset to be loaded.
        `n_steps` (`Optional[int]`): The number of steps in the dataset. If `None`, the default value of `51` is used.
        `coeff` (`Optional[float]`): The coefficient of the collision operator. If `None`, the default value of `1/area_of_unit_sphere(dimension)` is used.
    
    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: A tuple containing the loaded trajectory data and the corresponding parameters.
    """
    if n_steps is None:
        n_steps = 51
    if coeff is None:
        from    core.numerical.math   import  area_of_unit_sphere
        coeff = 1.0/area_of_unit_sphere(dimension)
    if isinstance(parts, int):
        parts = list(range(1, 1+parts))
    # dataset_path = Path(r"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE/parameterized_Boltzmann_solver/datasets/test") / f"Boltzmann_{dimension}D__coeff{coeff:.2e}__res_t{n_steps:03d}_v{resolution:03d}"
    dataset_path = _ROOT / "datasets" / "test" / f"Boltzmann_{dimension}D__coeff{coeff:.2e}__res_t{n_steps:03d}_v{resolution:03d}"
    
    from    itertools       import  product
    from    tqdm.notebook   import  tqdm
    trajectory = []
    parameters = []
    for alpha, part in tqdm(sorted(product(alpha, parts))):
        # if resolution<100:
        #     file_path = dataset_path / f"alpha{alpha:.2f}__part{part:02d}.pth"
        # else:
        file_path = dataset_path / f"alpha{alpha:.2f}__part{part:03d}.pth"
        try:
            loaded_file = torch.load(file_path, weights_only=False)
            _traj: torch.Tensor = loaded_file['data']
            _n_traj = _traj.size(0)
            trajectory.append(loaded_file['data'])
            parameters.append(loaded_file['vhs_alpha']*torch.ones((_n_traj, 1)))
        except FileNotFoundError:
            warnings.warn(f"File not found: {file_path}", UserWarning)
    trajectory = torch.cat(trajectory, dim=0)
    parameters = torch.cat(parameters, dim=0)
    assert trajectory.size(0)==parameters.size(0)
    n_traj = trajectory.size(0)
    print(f"Loaded {n_traj} trajectories from {dataset_path}.")
    
    trajectory = trajectory.type(torch.float).reshape(n_traj, n_steps, *([resolution]*dimension), 1)
    parameters = parameters.type(torch.float).reshape(n_traj, -1)
    return trajectory, parameters
    