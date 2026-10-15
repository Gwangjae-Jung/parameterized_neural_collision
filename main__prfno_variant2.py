import  argparse

import  sys
import  subprocess
from    pathlib             import  Path

from    itertools           import  product
from    ablation__config_variant2   import  *


MODEL_NAME      = "prfno"
BASE_PATH       = Path(__file__).parent / "step1__train_operators__variant2"
RUNFILE_PATH    = BASE_PATH / f"train__{MODEL_NAME}_variant2.py"


parser = argparse.ArgumentParser()
parser.add_argument("--cuda_index", type=int, help="The CUDA device to be used.")
args = parser.parse_args()


sys.stdout = open(BASE_PATH/f"{MODEL_NAME}{DIMENSION}D-param__ablation.log", "w")
for n_layers, max_freq in product(SET_N_LAYERS, SET_MAX_FREQ):    
    base_config = [
        "--seed",           str(SEED),
        "--num_epochs",     str(NUM_EPOCHS),
        "--batch_size",     str(BATCH_SIZE),
        "--learning_rate",  str(LEARNING_RATE),
        
        "--path_train",         str(PATH_TRAIN),
        "--path_val",           str(PATH_VAL),
        "--path_test",          str(None),
        "--exp_speed",          *map(str, EXP_SPEED),
        "--resolution",         str(RESOLUTION_TRAIN),
        "--train_part",         *map(str, TRAIN_PART),
        "--val_part",           *map(str, VAL_PART),
        "--test_part",          *map(str, TEST_PART),
        "--downsample",         str(DOWNSAMPLE),
        "--num_time_steps",     str(NUM_TIME_STEPS),
        
        "--model_type",         MODEL_NAME,
        "--dim_domain",         str(DIMENSION),
        "--max_freq",           str(max_freq),
        "--n_layers",           str(n_layers),
        "--in_channels",        str(IN_CHANNELS),
        "--hidden_channels",    str(HIDDEN_CHANNELS),
        "--out_channels",       str(OUT_CHANNELS),
        "--dim_parameters",     str(1),
        "--activation_name",    ACTIVATION_NAME,
    ]
    print(f"Running {MODEL_NAME.upper()}{DIMENSION}D with [{n_layers=}, {max_freq=}]", flush=True)
    subprocess.run(
        [
            "python", str(RUNFILE_PATH),
            "--cuda_index",     str(args.cuda_index),
        ] + base_config
    )