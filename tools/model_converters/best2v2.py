
import argparse
import torch
import einops
rearrange = einops.rearrange
import torch.nn.functional as F


new_model_data = {}

h = w = 14 
def convert_v2(args):

    data = torch.load(args.src)
    model_data = data['model']    
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

            value = F.interpolate(value,size=args.size,mode='bicubic')
            

            value = rearrange(
                value,
                ' b c h w -> b (h w) c',
            )

        new_key = 'decode_head.' + key
        


        new_model_data[new_key] = value
        
    torch.save(new_model_data, args.dst)


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'v2 style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    parser.add_argument('--size', help='interpolation size', default=(32,32))
    args = parser.parse_args()
    convert_v2(args)


if __name__ == '__main__':
    main()
