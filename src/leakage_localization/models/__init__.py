from .mlp_1d import MultilayerPerceptron_1d

_MODEL_CONSTRUCTORS = {
    'mlp-1d': MultilayerPerceptron_1d,
}
AVAILABLE_MODELS = list(_MODEL_CONSTRUCTORS.keys())

def load(name, **kwargs):
    if not(name in AVAILABLE_MODELS):
        raise NotImplementedError(f'Unrecognized model name: {name}.')
    model = _MODEL_CONSTRUCTORS[name](**kwargs)
    return model