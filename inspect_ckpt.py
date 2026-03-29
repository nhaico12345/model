import torch
import numpy as np

ckpt = torch.load('best_weather_model.pth', map_location='cpu', weights_only=False)

print('=== KEYS TRONG best_weather_model.pth ===')
for k in ckpt.keys():
    v = ckpt[k]
    if k == 'model_state' or k == 'model_state_dict':
        print(f'{k}: OrderedDict with {len(v)} keys')
    elif k == 'optimizer_state' or k == 'optimizer_state_dict':
        print(f'{k}: optimizer state (skipped)')
    elif k == 'scaler_state':
        print(f'{k}: {v}')
    elif k == 'scaler':
        sc = v
        print(f'scaler: {type(sc).__name__}')
        if hasattr(sc, 'mean_'):
            print(f'  mean_: {sc.mean_}')
        if hasattr(sc, 'scale_'):
            print(f'  scale_: {sc.scale_}')
        if hasattr(sc, 'feature_names_in_'):
            print(f'  features: {list(sc.feature_names_in_)}')
        if hasattr(sc, 'n_features_in_'):
            print(f'  n_features: {sc.n_features_in_}')
    elif k == 'config':
        print(f'\n=== CONFIG ===')
        for ck, cv in v.items():
            print(f'  {ck}: {cv}')
    elif k == 'feature_names':
        print(f'feature_names: {v}')
    else:
        print(f'{k}: {type(v).__name__} = {repr(v)[:200]}')
