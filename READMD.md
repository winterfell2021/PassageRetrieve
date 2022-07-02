## Pretrain model on MLM
** Works on RTX 3090 **
- roforme_v2_large: `python3.8 pretrain.py --config ./config/roformer_v2_large -b 8`
- roforme_v2_base: `python3.8 pretrain.py --config ./config/roformer_v2_base -b 16`
- roforme_v2_small: `python3.8 pretrain.py --config ./config/roformer_v2_small -b 32`

### Distributed Training
```bash
python3.8 -m torch.distributed.launch --nproc_per_node=2 pretrain.py --config ./config/roformer_v2_small -b 32
```