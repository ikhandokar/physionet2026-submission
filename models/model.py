import torch

from models.encoders import MLP, SignalEncoder
from models.fusion import FusionHead


class PhysioRiskModel(torch.nn.Module):
    def __init__(
        self,
        annotation_dim: int,
        metadata_dim: int,
        signal_embed_dim: int = 64,
        ann_embed_dim: int = 32,
        meta_embed_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.modalities = ["EEG", "ECG", "RESP", "EMG"]

        self.signal_encoders = torch.nn.ModuleDict({
            mod: SignalEncoder(embed_dim=signal_embed_dim)
            for mod in self.modalities
        })

        self.annotation_encoder = MLP(annotation_dim, ann_embed_dim, dropout=dropout)
        self.metadata_encoder = MLP(metadata_dim, meta_embed_dim, dropout=dropout)

        fusion_input_dim = len(self.modalities) * signal_embed_dim + ann_embed_dim + meta_embed_dim + len(self.modalities)
        self.fusion = FusionHead(fusion_input_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, batch):
        mask = batch["mask"]
        signal_embeddings = []

        for i, mod in enumerate(self.modalities):
            x = batch["signals"][mod]
            emb = self.signal_encoders[mod](x)
            emb = emb * mask[:, i:i + 1]
            signal_embeddings.append(emb)

        ann_emb = self.annotation_encoder(batch["annotations"])
        meta_emb = self.metadata_encoder(batch["metadata"])

        fused = torch.cat(signal_embeddings + [ann_emb, meta_emb, mask], dim=1)
        return self.fusion(fused).squeeze(1)