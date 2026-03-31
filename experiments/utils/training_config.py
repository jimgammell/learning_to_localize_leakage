from typing import List, Optional, Dict, Any, Annotated, Literal

from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator

from leakage_localization.datasets import DATASET
from leakage_localization.training.supervised_lightning_module import PREPROCESSING, LEAKAGE_MODEL
from leakage_localization.training.hyperparameter_tuning import ParamConfig
from leakage_localization.models.model import GREY_BOX_HEAD, TRUNK, POSITION_EMBEDDING, POOLING, HEAD, FNN_STYLE


class DataConfig(BaseModel):
    id: DATASET
    target_byte: List[int]
    target_variable: str
    preprocessing: PREPROCESSING
    random_roll_scale: Annotated[float, Field(ge=0)]
    random_lpf_scale: Annotated[float, Field(ge=0)]
    val_prop: Annotated[float, Field(gt=0, lt=1)]

    @field_validator('target_byte', mode='before')
    @classmethod
    def coerce_target_byte_to_list(cls, v: Any) -> Any:
        if isinstance(v, int):
            return [v]
        return v


class TrainingConfig(BaseModel):
    total_steps: Annotated[int, Field(gt=0)]
    lr_warmup_frac: Annotated[float, Field(ge=0, le=1)]
    lr_const_frac: Annotated[float, Field(ge=0, le=1)]
    batch_size: Annotated[int, Field(gt=0)]
    base_lr: Annotated[float, Field(gt=0)]
    lr_decay_multiplier: Annotated[float, Field(ge=0, le=1)]
    weight_decay: Annotated[float, Field(ge=0)]
    label_smoothing: Annotated[float, Field(ge=0, lt=1)]
    mixup_alpha: Annotated[float, Field(ge=0)]
    additive_gaussian_noise: Annotated[float, Field(ge=0)]
    grad_clip_val: Optional[Annotated[float, Field(gt=0)]] = None
    accumulate_grad_batches: Annotated[int, Field(gt=0)]
    early_stop_metric: str
    early_stop_mode: Literal['min', 'max']
    seed: int
    compile: StrictBool
    num_workers: Annotated[int, Field(ge=0)]


class MTDConfig(BaseModel):
    attack_count: Annotated[int, Field(gt=0)]
    traces_per_attack: Annotated[int, Field(gt=0)]


class ModelConfig(BaseModel):
    grey_box_head: Optional[GREY_BOX_HEAD]
    trunk: TRUNK
    position_embedding: POSITION_EMBEDDING
    pooling: POOLING
    head: HEAD
    fnn_style: FNN_STYLE
    patch_size: Optional[Annotated[int, Field(gt=0)]]
    use_fourier_embed: StrictBool
    fourier_embed_num_bands: Optional[Annotated[int, Field(gt=0)]]
    fourier_embed_sigma: Optional[Annotated[float, Field(gt=0)]]
    embedding_dim: Annotated[int, Field(gt=0)]
    expansion_factor: Annotated[int, Field(gt=0)]
    trunk_blocks: Annotated[int, Field(gt=0)]
    head_count: Optional[Annotated[int, Field(gt=0)]]
    register_tokens: Annotated[int, Field(ge=0)]
    input_dropout_rate: Annotated[float, Field(ge=0)]
    input_droppatch_rate: Annotated[float, Field(ge=0)]
    hidden_dropout_rate: Annotated[float, Field(ge=0)]
    use_bias: StrictBool
    perceiver_latent_dim: Optional[Annotated[int, Field(gt=0)]]
    perceiver_self_attn_per_cross_attn_blocks: Optional[Annotated[int, Field(gt=0)]]
    perceiver_cross_attn_head_count: Optional[Annotated[int, Field(gt=0)]]
    leakage_model: LEAKAGE_MODEL

    @model_validator(mode='after')
    def validate_cross_fields(self) -> 'ModelConfig':
        if self.use_fourier_embed:
            assert self.fourier_embed_num_bands is not None, 'fourier_embed_num_bands required when use_fourier_embed=True'
            assert self.fourier_embed_sigma is not None, 'fourier_embed_sigma required when use_fourier_embed=True'
        else:
            assert self.fourier_embed_num_bands is None, 'fourier_embed_num_bands must be null when use_fourier_embed=False'
            assert self.fourier_embed_sigma is None, 'fourier_embed_sigma must be null when use_fourier_embed=False'
        if self.trunk == 'perceiver':
            assert self.perceiver_latent_dim is not None, 'perceiver_latent_dim required when trunk=perceiver'
            assert self.perceiver_self_attn_per_cross_attn_blocks is not None, 'perceiver_self_attn_per_cross_attn_blocks required when trunk=perceiver'
            if self.perceiver_cross_attn_head_count is not None:
                assert self.perceiver_latent_dim % self.perceiver_cross_attn_head_count == 0, \
                    f'perceiver_latent_dim ({self.perceiver_latent_dim}) must be divisible by perceiver_cross_attn_head_count ({self.perceiver_cross_attn_head_count})'
        else:
            assert self.perceiver_latent_dim is None, 'perceiver_latent_dim must be null when trunk != perceiver'
            assert self.perceiver_self_attn_per_cross_attn_blocks is None, 'perceiver_self_attn_per_cross_attn_blocks must be null when trunk != perceiver'
            assert self.perceiver_cross_attn_head_count is None, 'perceiver_cross_attn_head_count must be null when trunk != perceiver'
        if self.head_count is not None:
            assert self.embedding_dim % self.head_count == 0, \
                f'embedding_dim ({self.embedding_dim}) must be divisible by head_count ({self.head_count})'
        return self


class SupervisedTrainingConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    mtd: MTDConfig
    model: ModelConfig
    search_space: Dict[str, Dict[str, ParamConfig]]
    commit_hash: Optional[str] = None