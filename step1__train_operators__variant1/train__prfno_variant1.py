##################################################
# Import libraries
##################################################
import  os, sys
from    typing              import  Union

import  time
from    pathlib             import  Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from    copy                import  deepcopy

import  numpy       as  np
import  torch
from    torch.utils.data            import  TensorDataset, DataLoader

from    core.utils                import  get_time_str
from    core.neural.utils         import  count_parameters, initialize_weights
from    core.utils                import  relative_error
from    variants        import  PRFNO_Variant1  as  PRFNO

from    train_utils     import  CollisionKeys
from    train_utils     import  uniform_grid
from    train_utils     import  load_data, exponential_cosine
from    train_utils     import  compute_loss_cons
from    train_utils     import  params_as_itself
from    train_utils     import  CustomParser
from    train_utils     import  sec_to_hms




##################################################
# Load the configuration file
##################################################
time_str        = get_time_str(False)
args            = CustomParser()
path_script     = Path(__file__).parent
DIMENSION       = args.dim_domain
RANDOM_SEED     = args.seed
SAMPLING_PERIOD = args.sampling_period

MODEL_TYPE  = args.model_type.upper()
MODEL_NAME  = f"{MODEL_TYPE}{DIMENSION}D"
NAME_OF_DIR = f"{MODEL_NAME}__n_layers{args.n_layers:02d}__max_freq{args.max_freq:02d}__seed{RANDOM_SEED}"
if SAMPLING_PERIOD>1:
    NAME_OF_DIR += f"__period{SAMPLING_PERIOD:02d}"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic  = True
torch.backends.cudnn.benchmark      = False

# Save the config files
BATCH_SIZE:     int             = args.batch_size
NUM_EPOCHS:     int             = args.num_epochs
TRAIN_SIZE:     int             = None
VAL_SIZE:       int             = None
LEARNING_RATE:  float           = args.learning_rate
DEVICE:         torch.device    = torch.device(f"cuda:{args.cuda_index}")

RESOLUTION:     int             = args.resolution
PATH_TRAIN:     Path            = args.path_train
PATH_VAL:       Path            = args.path_val
TRAIN_PART:     tuple[int]      = args.train_part
VAL_PART:       tuple[int]      = args.val_part

NUM_TIME_STEPS: int             = args.num_time_steps
EXP_SPEED:      tuple[float]    = args.exp_speed

PATH_BASE       = path_script / MODEL_NAME
PATH_CHECKPOINT = PATH_BASE / f"checkpoint__{NAME_OF_DIR}"
PATH_BEST_MODEL = PATH_BASE / NAME_OF_DIR
PATH_BASE.mkdir(exist_ok=True)
PATH_CHECKPOINT.mkdir(exist_ok=True)
PATH_BEST_MODEL.mkdir(exist_ok=True)

sys.stdout  = open(PATH_BASE/f"{NAME_OF_DIR}.log", "w")
cfg_print = {'sep': '\n', 'end': '\n\n', 'flush' : True,}
_pid_string     = f"- PID:        {os.getpid()}"
_ppid_string    = f"- Parent PID: {os.getppid()}"
_beginning_time_string = f"- Executed at {time.strftime('%Y-%m-%d %H:%M:%S')}"
_len_line = 80
_single_line = '-'*_len_line
_double_line = '='*_len_line
print(
    _double_line,
    _pid_string,
    _ppid_string,
    _beginning_time_string,
    _single_line,
    f"Training the model [{MODEL_NAME}] with",
    f"* n_layers: {args.n_layers}",
    f"* max_freq: {args.max_freq}",
    _single_line,
    f"* Random seed: {RANDOM_SEED}",
    f"* The device to be used: {torch.cuda.get_device_name()} ({DEVICE})",
    _double_line,
    **cfg_print,
)




##################################################
# Load data
##################################################
k_in:       CollisionKeys   = 'data'
k_param:    CollisionKeys   = 'params'
k_target:   CollisionKeys   = 'target'
train_data: dict[CollisionKeys, torch.Tensor]   = {}
val_data:   dict[CollisionKeys, torch.Tensor]   = {}


# Train data
_train_data, _train_info = load_data(PATH_TRAIN, EXP_SPEED, TRAIN_PART, sampling_period=SAMPLING_PERIOD)
MAX_V:  float   = _train_info['v_max']
train_data['data']      = _train_data['data']
train_data['params']    = params_as_itself(_train_data['exp_speed'])
train_data['target']    = _train_data['target']
TRAIN_SIZE      = train_data[k_in].size(0)
train_dataset   = TensorDataset(train_data['data'], train_data['params'], train_data['target'])
train_loader    = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Validation data
_val_data, _ = load_data(PATH_VAL, EXP_SPEED, VAL_PART)
val_data['data']        = _val_data['data']
val_data['params']      = params_as_itself(_val_data['exp_speed'])
val_data['target']      = _val_data['target']
VAL_SIZE    = val_data[k_in].size(0)
val_dataset = TensorDataset(val_data['data'], val_data['params'], val_data['target'])
val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

v_grid = uniform_grid(DIMENSION, RESOLUTION, MAX_V, device=DEVICE)
print(
    _double_line,
    f"- The size of the training dataset    >>> {TRAIN_SIZE}",
    f"- The shape of the training dataset   >>> {list(train_data[k_in].shape)}",
    _single_line,
    f"- The size of the validation dataset  >>> {VAL_SIZE}",
    f"- The shape of the validation dataset >>> {list(val_data[k_in].shape)}",
    _single_line,
    f"+ The maximum speed in each dimension >>> {MAX_V:.2f}",
    _double_line,
    **cfg_print,
)




##################################################
# Instantiate models, loss functions, optimizers, and learning-rate schedulers
##################################################
model_args = args.config_model(MODEL_TYPE.lower())
model = PRFNO(**model_args).to(DEVICE)
initialize_weights(model, "xavier normal")

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
lr_lambda = exponential_cosine(period=args.lr_period, half_life=float('inf'))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

print(
    _double_line,
    "Model configuration:",
    f"{model}",
    _single_line,
    f"Number of parameters in the model: {count_parameters(model)}",
    _double_line,
    **cfg_print,
)




##################################################
# Train the model
##################################################
train_history: dict[str, Union[list, float]] = {
    'train_rel_error':      [],
    'train_abs_error_cons': [],
    'val_rel_error':        [],
    'val_abs_error_cons':   [],
    'train_time':           0.0,
}

best_val_rel_error:         float   = None
best_val_abs_error_cons:    float   = None
best_model:         PRFNO           = None
best_model_idx:     int             = None

USED_TIME_STEPS:    int = 2

arr_coeff_cons = 0.5 * (1.0-torch.pow(0.9, torch.arange(NUM_EPOCHS, device=DEVICE)))

elapsed_time = time.time()
for epoch in range(1, NUM_EPOCHS+1):
    # NOTE: Train
    model.train()
    coeff_cons  = arr_coeff_cons[epoch-1]
    _train_time = time.time()
    train_rel_error:        float   = 0
    train_abs_error_cons:   float   = 0
    for data, params, target in train_loader:
    #########################
        data:   torch.Tensor = data.to(DEVICE)
        params: torch.Tensor = params.to(DEVICE)
        target: torch.Tensor = target.to(DEVICE)
        
        pred = model.forward(data, v_grid, params)
        # Loss 1 - Data-driven loss
        rel_error__data: torch.Tensor = relative_error(pred, target).sum()
        # Loss 2 - Conservation loss
        abs_error__cons: torch.Tensor = compute_loss_cons(pred, v_grid).sum()
        # Compute the total loss
        loss: torch.Tensor = rel_error__data + coeff_cons*abs_error__cons
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Backup
        train_rel_error         += rel_error__data.item()
        train_abs_error_cons    += abs_error__cons.item()
    #########################
    scheduler.step()
    _train_time = time.time() - _train_time
    train_rel_error         /= TRAIN_SIZE
    train_abs_error_cons    /= TRAIN_SIZE
    train_history['train_time'] += _train_time
    train_history['train_rel_error'].append(train_rel_error)
    train_history['train_abs_error_cons'].append(train_abs_error_cons)
    
    # NOTE: Validation
    model.eval()
    val_rel_error:      float   = 0
    val_abs_error_cons: float   = 0
    #########################
    with torch.inference_mode():
        for data, params, target in val_loader:
            data:   torch.Tensor = data.to(DEVICE)
            params: torch.Tensor = params.to(DEVICE)
            target: torch.Tensor = target.to(DEVICE)
            
            pred = model.forward(data, v_grid, params)
            # Loss 1 - Data-driven loss
            rel_error__data: torch.Tensor = relative_error(pred, target).sum()
            # Loss 2 - Conservation loss
            abs_error__cons: torch.Tensor = compute_loss_cons(pred, v_grid).sum()
            # Compute the total loss
            loss: torch.Tensor = rel_error__data + coeff_cons*abs_error__cons
            # Backup
            val_rel_error       += rel_error__data.item()
            val_abs_error_cons  += abs_error__cons.item()
    #########################            
    val_rel_error       /= VAL_SIZE
    val_abs_error_cons  /= VAL_SIZE
    train_history['val_rel_error'].append(val_rel_error)
    train_history['val_abs_error_cons'].append(val_abs_error_cons)
    
    # NOTE: Save the best model
    if (
            (best_val_rel_error is None or best_val_abs_error_cons is None) or 
            (best_val_rel_error+best_val_abs_error_cons > val_rel_error+val_abs_error_cons)
        ):
        best_val_rel_error      = val_rel_error
        best_val_abs_error_cons = val_abs_error_cons
        best_model      = deepcopy(model)
        best_model_idx  = epoch-1
        
    # NOTE: Report
    print(
        f"[ Epoch {epoch}/{NUM_EPOCHS} ] {train_history['train_time']/epoch:.1f}s/epoch",
        f"* coeff_cons: {coeff_cons:.4e} , lr: {scheduler.get_last_lr()[0]:.4e}",
        f"* Best model: epoch {best_model_idx+1} ({train_history['val_rel_error'][best_model_idx]:.4e} | {train_history['val_abs_error_cons'][best_model_idx]:.4e})",
        f"* [train]",
        f"  - Data loss (relative error)         {train_history['train_rel_error'][-1]:.4e}",
        f"  - Conservation loss (absolute error) {train_history['train_abs_error_cons'][-1]:.4e}",
        f"* [validation]",
        f"  - Data loss (relative error)         {train_history['val_rel_error'][-1]:.4e}",
        f"  - Conservation loss (absolute error) {train_history['val_abs_error_cons'][-1]:.4e}",
        **cfg_print,
    )
        
    if (
            best_val_rel_error < 1e-5 and
            train_history['val_abs_error_cons'][best_model_idx] < 1e-5
        ):
        print(f"Early stopping at epoch {epoch}.", **cfg_print)
        break
    
    if epoch%100==0:
        # Save the model
        __appendix__ = f"epoch{str(epoch).zfill(len(str(NUM_EPOCHS)))}"
        __name_checkpoint = f"state_dict__{__appendix__}.pth"
        model.cpu()
        best_model.cpu()
        torch.save(model.state_dict(), PATH_CHECKPOINT/__name_checkpoint)
        torch.save(best_model.state_dict(), PATH_CHECKPOINT/f"best__{__name_checkpoint}")
        model.to(DEVICE)
        best_model.to(DEVICE)
        torch.save(train_history, PATH_CHECKPOINT/f"train_history__{__appendix__}.pth")
        
elapsed_time = time.time() - elapsed_time
print(f"Elapsed time: {sec_to_hms(int(elapsed_time))}", **cfg_print)
model = best_model




##################################################
# Save the model and the train history
##################################################
torch.save(
    {
        'args':             model_args,
        'state_dict':       model.state_dict(),
        'train_history':    train_history,
    },
    PATH_BEST_MODEL/"model_and_history.pth",
)
print("Done.", flush=True)




##################################################
##################################################
# End of file