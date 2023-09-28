import argparse
import torch
import einops
rearrange = einops.rearrange
import torch.nn.functional as F

rearrange = einops.rearrange
new_model_data = {}

h = w = 14
def convert_baseline(args):
    pth = torch.load(args.src)

    for k, value in pth.items():


        if "patch_embed" in k:
            k = k.replace('proj','projection')
            new_k = 'decode_head.' + k

        elif "pos_embed" in k:
            value = value[:, 1:]
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
            new_k = 'decode_head.stages.0.' + k
        else:
            new_k = 'decode_head.stages.0.' + k
        new_model_data[new_k] = value
        torch.save(new_model_data, args.dst)


def main():
    parser = argparse.ArgumentParser(
        description='Convert ks in timm pretrained vit models to '
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
    args = parser.parse_args()
    convert_baseline(args)


if __name__ == '__main__':
    main()


