import models.modules.TOF_arch as TOF_arch
import torch
model = TOF_arch.TOFlow()
new_state = model.state_dict()

old_state = torch.load('../experiments/pretrained_models/TOF_official.pth')

count = 0
for (k_n, v_n), (k_o, v_o) in zip(new_state.items(), old_state.items()):
    print(k_n)
    if v_n.size() == v_o.size():
        new_state[k_n] = old_state[k_o]
        count += 1
    else:
        raise NotImplementedError
torch.save(new_state, '../experiments/pretrained_models/TOF_clean.pth')
