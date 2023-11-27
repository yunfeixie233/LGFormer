
import argparse
import torch
import einops
rearrange = einops.rearrange
import torch.nn.functional as F

from typing import Sequence
new_model_data = {}

h = w = 14

 
def convert_v2(args):

    data = torch.load(args.src)
    if 'model' in data.keys():
        model_data = data['model']
    elif 'state_dict' in data.keys():
        model_data = data['state_dict']
    for key, value in model_data.items():
        
        # print("ori",key,value.shape)
        if "pos_embed" in key:
            # value = value[:, 1:]
            value = rearrange(
                value,
                'b (h w) c -> b c h w',
                h=h,
                w=w,
            )

            value = F.interpolate(value,size=args.shape,mode='bicubic')
            

            value = rearrange(
                value,
                ' b c h w -> b (h w) c',
            )
        if args.group is not None:
            if args.group == [0]:
                if 'stages.0' in key:
                    branch_key = key.split('.',2)[-1]
                    branch_key = 'group_stages.0.' + branch_key
                    branch_key = 'decode_head.' + branch_key
                    new_model_data[branch_key] = value
            elif args.group == [0,1]:
                if 'stages.' in key:
                    branch_key = key.split('.',1)[-1]
                    branch_key = 'group_stages.' + branch_key
                    branch_key = 'decode_head.' + branch_key
                    new_model_data[branch_key] = value
            else:
                raise(ValueError(f"{args.group}not supported"))       
        new_key = 'decode_head.' + key
        # new_key = 'decode_head.' + key


        new_model_data[new_key] = value
        
    torch.save(new_model_data, args.dst)


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'v2 style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[32,32],
        help='targeted interpolated size')
    parser.add_argument(
        '--group',
        type=int,
        nargs='+',
        default=None,
        help='group stage')    
    args = parser.parse_args()
    convert_v2(args)


if __name__ == '__main__':
    main()
