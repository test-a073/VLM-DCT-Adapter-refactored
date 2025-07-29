import torch 
from typing import List, Dict


# --- Adapter Helper Functions (from main_chartQA.py) ---
def get_parent_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    names = name.split('.')
    parent = model
    for n in names[:-1]:
        parent = getattr(parent, n)
    return parent

def inject_adapters(
    model: torch.nn.Module,
    adapter_cls: type,
    base_adapter_args: dict, # Renamed from adapter_args for clarity
    layers_config: List[Dict[str, str]] # Expected format: [{'name': 'layer_name_pattern_to_match'}]
) -> torch.nn.Module:
    print(f"Starting adapter injection with {adapter_cls.__name__}...")
    for name, module in model.named_modules(): # Iterate over all module names in the model
        for layer_conf in layers_config:
            # Check if the current module's name matches the configuration name
            # The original code used 'in', which allows pattern matching.
            # If exact names are always provided in layer_conf, '==' could be used.
            # Sticking to 'in' to maintain original flexibility if patterns are used.
            if layer_conf['name'] in name:
                # Ensure we are matching the exact module intended, not a submodule containing the name
                # This check assumes layer_conf['name'] is the full name of the target module
                # if name == layer_conf['name']:
                print(f"Matched target layer for injection: {name}")
                try:
                    parent = get_parent_module(model, name)
                    original_module = getattr(parent, name.split('.')[-1])
                    
                    current_adapter_args = base_adapter_args.copy()

                    if hasattr(original_module, 'out_features') and isinstance(getattr(original_module, 'out_features'), int):
                        actual_in_features = original_module.out_features
                        print(f"Dynamically setting adapter 'in_features' for {name} to {actual_in_features} (derived from original_module.out_features).")
                        current_adapter_args['in_features'] = actual_in_features
                    else:
                        print(f"Original module {name} (type: {type(original_module)}) does not have an integer 'out_features' attribute. "
                                        f"Using 'in_features' from base_adapter_args: {current_adapter_args.get('in_features')}. "
                                        f"This might lead to errors if incorrect for this layer.")

                    adapter_instance = adapter_cls(**current_adapter_args)
                    setattr(parent, name.split('.')[-1], torch.nn.Sequential(original_module, adapter_instance))
                    print(f"Successfully injected adapter after {name} with args: {current_adapter_args}")
                except Exception as e:
                    print(f"Failed to inject adapter into {name}: {e}")
    return model
