from .multilayer_perceptron import MultilayerPerceptron
from .lenet import LeNet5
from .sca_cnn import SCA_CNN
from .resnet_1d import ResNet1d
from .mlp_1d import MultilayerPerceptron_1d
from .yap_cnn import YapCNN
from .perin_cnn import PerinCNN
from .transformer_for_sca import Transformer, TransformerConfig

_MODEL_CONSTRUCTORS = {
    'multilayer-perceptron': MultilayerPerceptron,
    'mlp-1d': MultilayerPerceptron_1d, 
    'lenet-5': LeNet5,
    'sca-cnn': SCA_CNN,
    'resnet-1d': ResNet1d,
    'yap-cnn': YapCNN,
    'perin-cnn': PerinCNN,
    'transformer': None
}
AVAILABLE_MODELS = list(_MODEL_CONSTRUCTORS.keys())

def load(name, **kwargs):
    if not(name in AVAILABLE_MODELS):
        raise NotImplementedError(f'Unrecognized model name: {name}.')
    if name == 'transformer':
        model = Transformer(TransformerConfig(**kwargs))
    else:
        model = _MODEL_CONSTRUCTORS[name](**kwargs)
    return model