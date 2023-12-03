
import argparse
from ast import arg
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
        if args.stage is not None:

                             
            if args.stage == [0]:
                if 'stages.1' in key:
                    branch_key = key.split('.',2)[-1]
                    branch_key = 'obj_stages.0.' + branch_key
                    branch_key = 'decode_head.' + branch_key
                    new_model_data[branch_key] = value
            elif args.stage == [0,1]:
                if 'sp_init' in key:
                    branch_key = key.split('.',1)[-1]
                    branch_key = 'obj_init.' +  branch_key
                    branch_key = 'decode_head.' + branch_key
                    new_model_data[branch_key] = value
                    
                if 'stem' in key:
                    branch_key = key.split('.',1)[-1]
                    branch_key = 'obj_stem.' +  branch_key
                    branch_key = 'decode_head.' + branch_key   
                    new_model_data[branch_key] = value                
                if 'stages.' in key:
                    branch_key = key.split('.',1)[-1]
                    branch_key = 'obj_stages.' + branch_key
                    branch_key = 'decode_head.' + branch_key
                    new_model_data[branch_key] = value
            else:
                raise(ValueError(f"{args.stage}not supported"))       
            
        if args.depth is not None:
            obj_idx = 10 - args.depth
            assert obj_idx >= 0 and obj_idx <= 9
            if 'stages.1.blocks' in key:
                parts = key.split('blocks.')
                
                # Further splitting the second part to isolate the number
                number_part = parts[1].split('.', 1)
                if int(number_part[0]) >= obj_idx:
                # Replacing the number with the new number
                    number_part[0] = str(int(number_part[0]) - obj_idx)
                # Reassembling the string with 'blocks_obj.'
                    branch_key = 'decode_head.' + parts[0] + 'blocks_obj.' + '.'.join(number_part)                
                    print(branch_key)
                    new_model_data[branch_key] = value
            
                
                
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
        '--stage',
        type=int,
        nargs='+',
        default=None,
        help='obj stage')    
    parser.add_argument(
        '--depth',
        type=int,
        default=None,
        help='obj depth')     
    args = parser.parse_args()
    convert_v2(args)


if __name__ == '__main__':
    main()
