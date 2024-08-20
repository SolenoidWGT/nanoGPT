import torch

# /fs-computility/llm/shared/llm-flash2.0/bin/python
# /fs-computility/llm/shared/wangguoteng/nanoGPT/cmp_ckpt.py
ckpt_folder = "./out-shakespeare-char"

import os
base_states = None
fns = os.listdir(ckpt_folder)
for i, fn in enumerate(fns):
    fn = os.path.join(ckpt_folder, fn)
    with open(fn, 'rb') as f:
        states = torch.load(f, map_location=torch.device('cpu'))
        if i == 0:
            base_states = states
        else:
            for key in base_states['model'].keys():
                assert torch.equal(base_states['model'][key], states['model'][key])

print("done")
