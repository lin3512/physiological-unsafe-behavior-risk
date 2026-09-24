"""Transformer definition, training loop, and probability inference."""
from __future__ import annotations


def train_transformer(x_train, y_train, x_val, y_val, cfg, device):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class RiskTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            dimension = cfg["d_model"]
            self.projection = nn.Linear(x_train.shape[-1], dimension)
            self.position = nn.Parameter(torch.zeros(1, cfg["sequence_length"], dimension))
            encoder_layer = nn.TransformerEncoderLayer(
                dimension, cfg["n_heads"], cfg["ff_dim"], cfg["dropout"],
                activation=lambda x: nn.functional.leaky_relu(x, .01), batch_first=True)
            decoder_layer = nn.TransformerDecoderLayer(
                dimension, cfg["n_heads"], cfg["ff_dim"], cfg["dropout"],
                activation=lambda x: nn.functional.leaky_relu(x, .01), batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, cfg["encoder_layers"])
            self.decoder = nn.TransformerDecoder(decoder_layer, cfg["decoder_layers"])
            self.start_token = nn.Parameter(torch.zeros(1, 1, dimension))
            self.output = nn.Linear(dimension, 1)

        def forward(self, x):
            memory = self.encoder(self.projection(x) + self.position[:, :x.shape[1]])
            token = self.start_token.expand(x.shape[0], -1, -1)
            decoded = self.decoder(token, memory)
            return self.output(decoded[:, 0]).squeeze(-1)

    model = RiskTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.float32)),
                        batch_size=cfg["batch_size"], shuffle=True)
    x_val_tensor = torch.tensor(x_val, device=device)
    best_loss, best_state, stale_epochs = float("inf"), None, 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=3)
    for _ in range(cfg["epochs"]):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_y_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
            val_loss = float(loss_fn(model(x_val_tensor), val_y_tensor).item())
        scheduler.step(val_loss)
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 10:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_probabilities(model, x, device):
    import torch
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(x, device=device))).cpu().numpy()
