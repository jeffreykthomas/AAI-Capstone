import torch
from . import model as llama_model


def load_model(check_point_path, map_location):
	checkpoint = torch.load(check_point_path, map_location=map_location)
	state_dict = checkpoint['model']
	model_config = checkpoint['model_config']
	model = llama_model.Transformer(model_config)
	unwanted_prefix = '_orig_mod.'
	for k, v in list(state_dict.items()):
		if k.startswith(unwanted_prefix):
			state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

	model.load_state_dict(state_dict)
	return model