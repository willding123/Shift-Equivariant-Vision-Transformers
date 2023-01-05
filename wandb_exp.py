#%% 
import wandb
import torch 
from models.swin_transformer_poly import PolySwin
import torch.nn.functional as F
from config import get_config
import argparse

parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
# parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
# parser.add_argument('--data-path', type=str, help='path to dataset')
# parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
# parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
#                     help='no: no cache, '
#                             'full: cache all data, '
#                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
# parser.add_argument('--pretrained',
#                     help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
# parser.add_argument('--resume', help='resume from checkpoint')
# parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
# parser.add_argument('--use-checkpoint', action='store_true',
#                     help="whether to use gradient checkpointing to save memory")
# parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
# parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
#                     help='mixed precision opt level, if O0, no amp is used (deprecated!)')
# parser.add_argument('--output', default='output', type=str, metavar='PATH',
#                     help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
# parser.add_argument('--tag', help='tag of experiment')
# parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
# parser.add_argument('--throughput', action='store_true', help='Test throughput only')

# # distributed training
# parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

# # for acceleration
# parser.add_argument('--fused_window_process', action='store_true',
#                     help='Fused window shift & window partition, similar for reversed part.')
# parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
# ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
# parser.add_argument('--optim', type=str,
#                     help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

args, unparsed = parser.parse_known_args()
args.cfg = "configs/swin/swin_tiny_patch4_window7_224_22k.yaml"
args.local_rank = 0
config = get_config(args)
wandb.init(config=config)

model = PolySwin((224,224))

# Magic
wandb.watch(model, log_freq=100)

model.train()

output = model(torch.rand((1,3,224,224)))
loss = F.nll_loss(output, target)
loss.backward()
optimizer.step()
wandb.log({"loss": loss})
# %%
