import torch.nn as nn
import torch.nn.functional as F


class LOBCNN(nn.Module):
    """
    Simple DeepLOB-style CNN baseline for LOB sequences.

    Expects input x of shape [batch_size, seq_len, input_dim].
    Internally reshapes to [batch_size, 1, input_dim, seq_len] and
    applies a few Conv2d blocks, then global average pooling and an MLP head.
    """

    def __init__(self, config):
        super().__init__()

        self.seq_length = config.seq_length
        self.input_dim = config.input_dim
        self.num_classes = config.num_classes
        self.dropout_p = getattr(config, "dropout", 0.1)

        # You can tweak these if you like (more/less channels)
        c1, c2, c3 = 32, 64, 128

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=c1,
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
            nn.ReLU(inplace=True),
            # pool only along time axis (width); keep feature resolution
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=c1,
                out_channels=c2,
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=c2,
                out_channels=c3,
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
            nn.ReLU(inplace=True),
            # optional: one more pooling along time; you can remove if seq_len is small
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        # After the conv blocks, we will do global average pooling over (H, W),
        # so the feature dimension becomes just c3.
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc1 = nn.Linear(c3, 128)
        self.fc_out = nn.Linear(128, self.num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        returns logits: [batch_size, num_classes]
        """
        # Rearrange to [B, 1, H=input_dim, W=seq_len]
        # Dataset gives x as [B, T, F]; treat features as "height", time as "width"
        x = x.permute(0, 2, 1)  # [B, F, T]
        x = x.unsqueeze(1)  # [B, 1, F, T]

        x = self.conv_block1(x)  # [B, c1, H, W']
        x = self.conv_block2(x)  # [B, c2, H, W'']
        x = self.conv_block3(x)  # [B, c3, H, W''']

        # Global average pooling over spatial dims (H, W)
        x = x.mean(dim=(2, 3))  # [B, c3]

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc_out(x)  # [B, num_classes]

        return logits
