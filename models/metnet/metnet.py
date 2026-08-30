import torch
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
from huggingface_hub import PyTorchModelHubMixin

from models.metnet.layers.ConditionTime import ConditionTime
from models.metnet.layers.ConvGRU import ConvGRU
from models.metnet.layers.DownSampler import DownSampler
from models.metnet.layers.Preprocessor import MetNetPreprocessor
from models.metnet.layers.TimeDistributed import  TimeDistributed

from utils.registry import register_module

@register_module(parent="models")
class MetNet(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        image_encoder: str = "downsampler",
        input_channels: int = 12,
        input_size: int = 256,
        output_channels: int = 12,
        hidden_dim: int = 2048,
        kernel_size: int = 3,
        num_layers: int = 1,
        num_att_layers: int = 2,
        num_att_heads: int = 16,
        forecast_steps: int = 24,
        temporal_dropout: float = 0.2,
        use_preprocessor: bool = True,
        low_mem: bool = False,
        **kwargs,
    ):
        super(MetNet, self).__init__()
        config = locals()
        config.pop("self")
        config.pop("__class__")
        self.config = kwargs.pop("config", config)
        input_size = self.config["input_size"]
        input_channels = self.config["input_channels"]
        temporal_dropout = self.config["temporal_dropout"]
        image_encoder = self.config["image_encoder"]
        forecast_steps = self.config["forecast_steps"]
        hidden_dim = self.config["hidden_dim"]
        kernel_size = self.config["kernel_size"]
        num_layers = self.config["num_layers"]
        num_att_layers = self.config["num_att_layers"]
        output_channels = self.config["output_channels"]
        use_preprocessor = self.config["use_preprocessor"]
        num_att_heads = self.config.get("num_att_heads", 16)
        low_mem = self.config.get("low_mem", False)

        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.drop = nn.Dropout(temporal_dropout)
        if image_encoder in ["downsampler", "default"]:
            image_encoder = DownSampler(input_channels + forecast_steps)
        else:
            raise ValueError(f"Image_encoder {image_encoder} is not recognized")
        self.image_encoder = TimeDistributed(image_encoder, low_mem=low_mem)
        self.ct = ConditionTime(forecast_steps) # 对时间进行ont-hot编码，forecast_steps表示编码长度
        self.temporal_enc = TemporalEncoder(
            image_encoder.output_channels,
            hidden_dim,
            ks=kernel_size,
            n_layers=num_layers,
            use_checkpoint=low_mem,
        )
        self.position_embedding = AxialPositionalEmbedding(
            dim=self.temporal_enc.out_channels, shape=(input_size, input_size)
        )
        self.temporal_agg = nn.Sequential(
            *[
                AxialAttention(dim=hidden_dim, dim_index=1, heads=num_att_heads, num_dimensions=2)
                for _ in range(num_att_layers)
            ]
        )

        self.head = nn.Conv2d(hidden_dim, output_channels, kernel_size=(1, 1))  # Reduces to mask

    def encode_timestep(self, x, fstep=1):
        # Condition Time
        x = self.ct(x, fstep)
        ##CNN
        x = self.image_encoder(x)
        # Temporal Encoder
        state = self.temporal_enc(self.drop(x))
        # AxialAttention is numerically unstable under fp16 autocast and can
        # crash autograd with a bare SystemError. Keep this block in fp32.
        with torch.cuda.amp.autocast(enabled=False):
            state = state.float()
            state = self.position_embedding(state)
            state = self.temporal_agg(state)
        return state

    def forward(self, imgs: torch.Tensor, lead_time: int = 0) -> torch.Tensor:
        """It takes a rank 5 tensor
        - imgs [bs, seq_len, channels, h, w]
        """
        x_i = self.encode_timestep(imgs, lead_time)
        res = self.head(x_i)
        return res


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=384, ks=3, n_layers=1, use_checkpoint=False):
        super().__init__()
        self.out_channels = out_channels
        self.rnn = ConvGRU(
            in_channels, out_channels, (ks, ks), n_layers, batch_first=True, use_checkpoint=use_checkpoint
        )

    def forward(self, x):
        _, h = self.rnn(x, return_sequence=False)
        return h[-1]


def feat2image(x, target_size=(128, 128)):
    "This idea comes from MetNet"
    x = x.transpose(1, 2)
    return x.unsqueeze(-1).unsqueeze(-1) * x.new_ones(1, 1, 1, *target_size)

if __name__ == "__main__":
    import random

    import torch.nn.functional as F

    def print_gpu_mem(tag: str):
        if not torch.cuda.is_available():
            print(f"[显存] {tag}: (无 CUDA，跳过)")
            return
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[显存] {tag}: allocated={alloc:.0f} MB, reserved={reserved:.0f} MB, peak={peak:.0f} MB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4  # 11G 显存建议从 1 开始，稳定后再逐步增大
    forecast_steps = 96 # 预测的未来时间步数
    forecast_steps_train = 32 # 训练时使用的预测未来时间步数
    bin_nums = 512 # 预测的bin数量（分辨率：0.2 mm/h）
    hidden_dim = 1024 # 隐藏层维度
    input_size = 256 # 输入图像大小
    seq_len = 16 # 输入历史数据长度
    num_layers = 6 # 卷积GRU层数
    num_att_layers = 6 # 轴注意力层数


    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = MetNet(
        hidden_dim=hidden_dim,
        forecast_steps=forecast_steps,
        input_channels=1,
        output_channels=bin_nums,
        input_size=int(input_size / 4),
        num_layers=num_layers,
        num_att_layers=num_att_layers,
        low_mem=True,
    ).to(device)
    print(model)
    param_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024**2
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M ({param_mb:.0f} MB)")
    print_gpu_mem("模型加载后")

    # MetNet expects original HxW to be 4x the input size
    x = torch.randn((batch_size, seq_len, 1, input_size, input_size), device=device)
    lead_times = random.sample(range(forecast_steps), k=forecast_steps_train)
    out_h, out_w = input_size // 4, input_size // 4

    # 训练测试：每个 lead_time 单独 backward，避免 96 路计算图同时驻留显存
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for i, lead_time in enumerate(lead_times):
        target = torch.randn((batch_size, bin_nums, out_h, out_w), device=device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x, lead_time)
            loss = F.mse_loss(pred, target) / len(lead_times)
        scaler.scale(loss).backward()
        total_loss += loss.item()
        print(f"  lead_time={lead_time:2d}, loss={loss.item():.4f}")
        print_gpu_mem("backward 后")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scaler.step(optimizer)
    scaler.update()
    print(f"训练平均 loss: {total_loss:.4f}")
    print_gpu_mem("optimizer.step 后")