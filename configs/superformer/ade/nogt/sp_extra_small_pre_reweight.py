_base_ = [
    './sp_extra_small_pre.py'
]
model = dict(
    decode_head=dict(
        reweight = True
))
