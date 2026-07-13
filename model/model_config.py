"""Model-specific configurations (decoupled from model definitions)."""


class Qwen3_5Config:
    model_type = "qwen3_5"

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 1024,
        intermediate_size: int = 3584,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        max_position_embeddings: int = 262144,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attn_output_gate: bool = True,
        tie_word_embeddings: bool = True,
        hidden_act: str = "silu",
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 16,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        rope_theta: float = 10000000.0,
        partial_rotary_factor: float = 0.25,
        layer_types: list[str] | None = None,
        mamba_ssm_dtype: str = "float32",
        vision_config=None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        linear_attn_prefill_backend: str = "torch",
        linear_attn_decode_backend: str = "fla",
        **kwargs,
    ):
        if layer_types is None:
            layer_types = [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attn_output_gate = attn_output_gate
        self.hidden_act = hidden_act
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.layer_types = layer_types
        self.mamba_ssm_dtype = mamba_ssm_dtype
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.linear_attn_prefill_backend = linear_attn_prefill_backend
        self.linear_attn_decode_backend = linear_attn_decode_backend
