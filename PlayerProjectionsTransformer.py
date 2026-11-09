"""
PlayerProjectionsTransformer.py
=================================
Hugging Face `transformers`-based counterpart to PlayerProjections.py's
PlayerStatLSTM -- same task (predict a player's next-game stat line from
their last WINDOW_SIZE games plus this week's matchup context), same data
pipeline (100% reused from PlayerProjections.py -- only the model
architecture and its train/save/load/predict wrappers are new here), but a
self-attention Transformer encoder over the game sequence instead of an
LSTM, built on transformers.PreTrainedModel so it gets Hugging Face Hub
support (save_pretrained/from_pretrained/push_to_hub) for free.

WHY A SEPARATE FILE/MODEL, NOT A REPLACEMENT
-----------------------------------------------
There is no existing pretrained checkpoint on the Hub for NFL stat-line
prediction, so this model gets no pretraining benefit over PlayerStatLSTM --
it's trained from scratch on exactly the same data either way. This file
exists to (a) make the model shareable/versionable through the Hub's
standard tooling and (b) have a second architecture to compare against, not
because the Transformer is known to predict better. PlayerProjections.py's
LSTM pipeline is completely untouched by any of this -- every function here
is new, only the data-prep helpers (build_matchup_context(), build_sequences(),
Normalizer, etc.) are imported and reused as-is.

ARCHITECTURE
------------
Mirrors PlayerStatLSTM's design (see that class's docstring): a sequence
encoder processes the window of past games, matchup context is projected
through its own small encoder rather than concatenated raw (same reasoning
about giving context equal standing applies here), and one linear head
predicts the next game's stat vector from [sequence representation,
context embedding]. The only real difference is the sequence encoder
itself -- a Transformer encoder (self-attention across the window's game
steps, with a learned per-position embedding since the window is a small,
fixed length) instead of a stacked LSTM. The final window position's
contextualized representation (which self-attention lets attend to every
earlier game) plays the same role PlayerStatLSTM's final hidden state does.

WHAT'S DIFFERENT FROM PlayerProjections.py's SAVE/LOAD
-----------------------------------------------------------
PlayerStatLSTM's raw PyTorch state dict carries no metadata, so save_model()
has to hand-write a sidecar _meta.json with stat_columns/normalizer stats/
window_size/etc. A Hugging Face PretrainedConfig serializes any attribute
set on it to config.json automatically, so PlayerStatTransformerConfig just
carries all of that metadata as regular config fields -- save_pretrained()/
from_pretrained() (both inherited from PreTrainedModel, no code written
here) handle weights + metadata together, no separate sidecar file needed.

USAGE
-----
    python PlayerProjectionsTransformer.py
        Trains and saves all four position models (into ./models_transformer/).

    from PlayerProjectionsTransformer import train_all_position_models_hf, predict_next_game_hf
    train_all_position_models_hf()
    predict_next_game_hf(player_id="00-0039732", next_opponent="LV")

    # Push/pull through the Hub directly -- PreTrainedModel already provides
    # this (PushToHubMixin), no extra wrapper code needed here:
    from PlayerProjectionsTransformer import load_model_hf, PlayerStatTransformerModel
    model = load_model_hf("QB")
    model.push_to_hub("yourname/nfl-qb-transformer")   # needs `hf auth login` first
    # ...later, from anywhere:
    # PlayerStatTransformerModel.from_pretrained("yourname/nfl-qb-transformer")
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PretrainedConfig

from PlayerProjections import (
    position_training_rows,
    POSITION_STAT_COLUMNS,
    CONTEXT_COLUMNS,
    WINDOW_SIZE,
    POISSON_WEIGHT,
    load_weekly_stats,
    load_defensive_target_stats,
    load_player_bio,
    load_schedule_context,
    load_player_age,
    load_injury_reports,
    build_top_cb_profile,
    build_matchup_context,
    compute_current_opponent_tiers,
    build_sequences,
    Normalizer,
    prepare_position_training_data,
    PlayerGameSequenceDataset,
    _predict_next_game_core,
)

TRANSFORMER_MODEL_DIR = "models_transformer"  # kept separate from PlayerProjections.py's MODEL_DIR ("models") -- different architecture, different files, no collision


# ---------------------------------------------------------------------------
# CONFIG -- carries architecture sizing AND everything PlayerProjections.py's
# save_model() has to hand-write into a separate _meta.json for the LSTM
# ---------------------------------------------------------------------------
class PlayerStatTransformerConfig(PretrainedConfig):
    """Everything needed to reconstruct and use a trained PlayerStatTransformerModel -- see module docstring."""

    model_type = "player_stat_transformer"

    def __init__(
        self,
        num_features: int = 1,
        context_size: int = 0,
        window_size: int = WINDOW_SIZE,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        context_embed_size: int = 24,
        stat_columns: Optional[List[str]] = None,
        context_columns: Optional[List[str]] = None,
        normalizer_mean: Optional[List[float]] = None,
        normalizer_std: Optional[List[float]] = None,
        context_normalizer_mean: Optional[List[float]] = None,
        context_normalizer_std: Optional[List[float]] = None,
        context_fill_values: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        self.num_features = num_features
        self.context_size = context_size
        self.window_size = window_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.context_embed_size = context_embed_size if context_size > 0 else 0
        self.stat_columns = stat_columns or []
        self.context_columns = context_columns or []
        self.normalizer_mean = normalizer_mean
        self.normalizer_std = normalizer_std
        self.context_normalizer_mean = context_normalizer_mean
        self.context_normalizer_std = context_normalizer_std
        self.context_fill_values = context_fill_values or {}
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
class PlayerStatTransformerModel(PreTrainedModel):
    """
    Transformer counterpart to PlayerProjections.PlayerStatLSTM -- same
    inputs/outputs, a self-attention encoder over the game sequence instead
    of an LSTM. See this module's docstring for the full design reasoning.
    """

    config_class = PlayerStatTransformerConfig

    def __init__(self, config: PlayerStatTransformerConfig):
        super().__init__(config)
        self.input_proj = nn.Linear(config.num_features, config.d_model)
        self.position_embed = nn.Parameter(torch.zeros(1, config.window_size, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.nhead, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        if config.context_size > 0:
            self.context_encoder = nn.Sequential(nn.Linear(config.context_size, config.context_embed_size), nn.ReLU())
        else:
            self.context_encoder = None

        self.head = nn.Linear(config.d_model + config.context_embed_size, config.num_features)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            x: (batch, window_size, num_features)
            context: (batch, context_size) -- required whenever
                context_size > 0 (every model this file trains).
            labels: (batch, num_features) -- if given, MSE loss against
                `x`'s predicted next-game vector is returned too (the same
                "reward for closeness" signal PlayerStatLSTM's training
                loop uses -- see PlayerProjections.py's module docstring).

        Returns:
            {"predictions": (batch, num_features)} plus "loss" if `labels`
            was given.
        """
        h = self.input_proj(x) + self.position_embed[:, : x.shape[1], :]
        h = self.encoder(h)
        last_step = h[:, -1, :]  # final window position's representation -- has attended to every earlier game, playing the same role PlayerStatLSTM's final hidden state does

        if self.context_encoder is not None:
            if context is None:
                raise ValueError("This model was trained with matchup context features; forward() requires `context`.")
            context_embedding = self.context_encoder(context)
            last_step = torch.cat([last_step, context_embedding], dim=1)

        predictions = self.head(last_step)
        out = {"predictions": predictions}
        if labels is not None:
            out["loss"] = nn.functional.mse_loss(predictions, labels)
        return out


# ---------------------------------------------------------------------------
# TRAINING -- same data prep as PlayerProjections.train_position_model(), reused as-is
# ---------------------------------------------------------------------------
def train_position_model_hf(
    position_group: str,
    df: pd.DataFrame,
    context_df: pd.DataFrame,
    top_cb_profile: pd.DataFrame,
    player_bio: pd.DataFrame,
    player_age: pd.DataFrame,
    injury_reports: pd.DataFrame,
    archetypes: pd.DataFrame,
    defense_vs_archetype: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    device: Optional[str] = None,
) -> Tuple[PlayerStatTransformerModel, List[dict]]:
    """
    Train one PlayerStatTransformerModel for a single position group -- the
    Transformer counterpart to PlayerProjections.train_position_model(). All
    data prep (leak-safe context building, sequence construction,
    chronological split, normalization) is identical and reused directly
    from PlayerProjections.py; only the model architecture and training
    loop are new. See train_position_model()'s docstring for the full
    leak-safety/normalization reasoning -- unchanged here.

    Args:
        position_group, df, context_df, top_cb_profile, player_bio,
            player_age, injury_reports, archetypes, defense_vs_archetype,
            window_size, epochs, batch_size, lr,
            val_frac: Same meaning as train_position_model().
        d_model, nhead, num_layers: Transformer encoder sizing (see
            PlayerStatTransformerModel) -- d_model must be divisible by nhead.
        device: "cpu"/"cuda"; autodetects if not given.

    Returns:
        (model, history) -- model is a ready-to-save PlayerStatTransformerModel
        with stat_columns/normalizer/context_normalizer/context_fill_values
        already baked into model.config (see save_model_hf() -- unlike
        PlayerStatLSTM, no separate sidecar file is needed). history is the
        same per-epoch {"epoch", "train_loss", "val_loss"} list
        train_position_model() returns.

    Raises:
        ValueError: same conditions as train_position_model() (unknown
            position_group, or fewer than 20 training examples).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    (
        stat_columns, X_train, C_train, y_train, X_val, C_val, y_val,
        normalizer, context_normalizer, context_fill_values, weights_train, weights_val,
    ) = prepare_position_training_data(
        position_group, df, context_df, top_cb_profile, player_bio, player_age,
        injury_reports, archetypes, defense_vs_archetype, window_size, val_frac,
    )

    train_ds = PlayerGameSequenceDataset(
        normalizer.transform(X_train), context_normalizer.transform(C_train), normalizer.transform(y_train), weights_train
    )
    val_ds = PlayerGameSequenceDataset(
        normalizer.transform(X_val), context_normalizer.transform(C_val), normalizer.transform(y_val), weights_val
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    config = PlayerStatTransformerConfig(
        num_features=len(stat_columns), context_size=len(CONTEXT_COLUMNS), window_size=window_size,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        stat_columns=stat_columns, context_columns=CONTEXT_COLUMNS,
        normalizer_mean=normalizer.mean.tolist(), normalizer_std=normalizer.std.tolist(),
        context_normalizer_mean=context_normalizer.mean.tolist(), context_normalizer_std=context_normalizer.std.tolist(),
        context_fill_values=context_fill_values,
    )
    model = PlayerStatTransformerModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Mixed loss: MSE plus an auxiliary Poisson term for count-like columns
    # (touchdowns/interceptions) -- same reasoning and implementation as
    # PlayerProjections.train_position_model()'s Notes (computed on
    # denormalized predictions via the same affine Normalizer already fit
    # for MSE, so no separate output head or normalization path needed).
    # Loss is computed here rather than inside PlayerStatTransformerModel.forward()
    # (which only knows plain MSE) so both ensemble members -- this one and
    # PlayerStatLSTM -- train on identical loss logic.
    poisson_idx = [i for i, c in enumerate(stat_columns) if c.endswith("_tds") or c.endswith("_interceptions")]
    mean_t = torch.tensor(normalizer.mean, device=device)
    std_t = torch.tensor(normalizer.std, device=device)

    def compute_loss(preds: torch.Tensor, yb: torch.Tensor, wb: torch.Tensor) -> torch.Tensor:
        per_example_mse = ((preds - yb) ** 2).mean(dim=1)
        loss = (per_example_mse * wb).sum() / wb.sum()
        if poisson_idx:
            pred_real = torch.clamp(preds[:, poisson_idx] * std_t[poisson_idx] + mean_t[poisson_idx], min=1e-3)
            target_real = yb[:, poisson_idx] * std_t[poisson_idx] + mean_t[poisson_idx]
            per_example_poisson = (pred_real - target_real * torch.log(pred_real)).mean(dim=1)
            poisson_nll = (per_example_poisson * wb).sum() / wb.sum()
            loss = loss + POISSON_WEIGHT * poisson_nll
        return loss

    # Same best-val-loss checkpointing as train_position_model() -- these
    # models overfit fast on a small dataset, so the final epoch isn't
    # necessarily the best one (see that function's Notes).
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, cb, yb, wb in train_loader:
            xb, cb, yb, wb = xb.to(device), cb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model(xb, cb)["predictions"], yb, wb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        train_loss = running_loss / max(len(train_ds), 1)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for xb, cb, yb, wb in val_loader:
                xb, cb, yb, wb = xb.to(device), cb.to(device), yb.to(device), wb.to(device)
                val_running_loss += compute_loss(model(xb, cb)["predictions"], yb, wb).item() * len(xb)
        val_loss = val_running_loss / max(len(val_ds), 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"[{position_group}-transformer] epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[{position_group}-transformer] using epoch {best_epoch}'s weights (lowest val_loss={best_val_loss:.4f})")

    return model, history


# ---------------------------------------------------------------------------
# SAVE / LOAD -- standard Hugging Face save_pretrained/from_pretrained, no
# hand-written sidecar needed (see module docstring)
# ---------------------------------------------------------------------------
def save_model_hf(model: PlayerStatTransformerModel, position_group: str, model_dir: str = TRANSFORMER_MODEL_DIR) -> str:
    """
    Save a trained PlayerStatTransformerModel the standard Hugging Face way.

    Args:
        model: A trained PlayerStatTransformerModel.
        position_group: "QB"/"RB"/"WR"/"TE" -- used to name the output subdirectory.
        model_dir: Directory to save into (created if missing).

    Returns:
        The directory saved into (<model_dir>/<position_group.lower()>).
    """
    path = os.path.join(model_dir, position_group.lower())
    model.save_pretrained(path)
    print(f"Saved {position_group} transformer model -> {path}/")
    return path


def load_model_hf(position_group: str, model_dir: str = TRANSFORMER_MODEL_DIR, device: Optional[str] = None) -> PlayerStatTransformerModel:
    """
    Load a previously saved position transformer model, ready for
    predict_next_game_hf() (or model.push_to_hub()).

    Args:
        position_group: "QB"/"RB"/"WR"/"TE".
        model_dir: Directory save_model_hf() wrote into, OR a Hub repo id
            (from_pretrained() handles both transparently).
        device: "cpu"/"cuda"; autodetects if not given.

    Returns:
        The model, already in eval() mode and on `device`.

    Raises:
        FileNotFoundError: if this position group hasn't been trained/saved
            yet locally -- run train_all_position_models_hf() first.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(model_dir, position_group.lower())
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"No saved transformer model for {position_group!r} in {path}/ -- run "
            f"train_all_position_models_hf() first."
        )
    model = PlayerStatTransformerModel.from_pretrained(path).to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# ENTRY POINT (train once on all history)
# ---------------------------------------------------------------------------
def train_all_position_models_hf(years: Optional[List[int]] = None, model_dir: str = TRANSFORMER_MODEL_DIR,
                                 starters_only: bool = False, **train_kwargs) -> Dict[str, List[dict]]:
    """
    Train and save a transformer model for every position group in
    POSITION_STAT_COLUMNS -- the Transformer counterpart to
    PlayerProjections.train_all_position_models().

    Args:
        years: Passed through to load_weekly_stats() (defaults to the last
            3 seasons if not given).
        model_dir: Directory to save each position's model into.
        starters_only: QB only -- train the QB model on listed starters' games
            (see position_training_rows()); other positions are unaffected.
            Defaults to False, the original behaviour.
        **train_kwargs: Passed through to train_position_model_hf() for
            every position group (e.g. epochs=100, d_model=128).

    Returns:
        Dict mapping position_group -> its training history.
    """
    from PlayerArchetypes import compute_all_archetypes, compute_all_defense_vs_archetype_profiles

    df = load_weekly_stats(years=years)
    defensive_target_stats = load_defensive_target_stats(years=years)
    player_bio = load_player_bio()
    schedule_context = load_schedule_context(years=years)
    player_age = load_player_age(years=years)
    injury_reports = load_injury_reports(years=years)
    top_cb_profile = build_top_cb_profile(defensive_target_stats, player_bio)
    context_df = build_matchup_context(df, top_cb_profile, schedule_context, injury_reports, player_bio)
    archetypes_by_position = compute_all_archetypes(df)
    defense_vs_archetype_by_position = compute_all_defense_vs_archetype_profiles(df, archetypes_by_position)

    results = {}
    for position_group in POSITION_STAT_COLUMNS:
        print(f"\n=== Training {position_group} transformer model ===")
        model, history = train_position_model_hf(
            position_group, position_training_rows(df, schedule_context, position_group, starters_only), context_df, top_cb_profile, player_bio, player_age, injury_reports,
            archetypes_by_position[position_group], defense_vs_archetype_by_position[position_group], **train_kwargs
        )
        # Recorded on the config, which save_pretrained() writes to config.json,
        # so predict_next_game_hf() feeds the model the history it trained on.
        model.config.starters_only = starters_only and position_group == "QB"
        save_model_hf(model, position_group, model_dir)
        results[position_group] = history

    return results


# ---------------------------------------------------------------------------
# PREDICTION (using an already-trained/saved model)
# ---------------------------------------------------------------------------
def predict_next_game_hf(
    player_id: str,
    next_opponent: str,
    df: Optional[pd.DataFrame] = None,
    defensive_target_stats: Optional[pd.DataFrame] = None,
    player_bio: Optional[pd.DataFrame] = None,
    schedule_context: Optional[pd.DataFrame] = None,
    player_age: Optional[pd.DataFrame] = None,
    injury_reports: Optional[pd.DataFrame] = None,
    injury_status: Optional[str] = None,
    archetypes: Optional[pd.DataFrame] = None,
    defense_vs_archetype: Optional[pd.DataFrame] = None,
    position_group: Optional[str] = None,
    model_dir: str = TRANSFORMER_MODEL_DIR,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Transformer counterpart to PlayerProjections.predict_next_game() -- same
    contract and behavior (see that function's full docstring), just loads
    and runs a PlayerStatTransformerModel instead of a PlayerStatLSTM.
    Reads the normalizer/context_fill_values/stat_columns/window_size this
    needs straight off the loaded model's config (see module docstring)
    rather than a separate meta.json. `injury_status` overrides whatever's
    in `injury_reports` for this player's upcoming week, same meaning as
    predict_next_game()'s.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(pg):
        model = load_model_hf(pg, model_dir, device)
        config = model.config
        normalizer = Normalizer(np.array(config.normalizer_mean, dtype=np.float32), np.array(config.normalizer_std, dtype=np.float32))
        context_normalizer = Normalizer(np.array(config.context_normalizer_mean, dtype=np.float32), np.array(config.context_normalizer_std, dtype=np.float32))
        return (model, config.stat_columns, config.window_size, normalizer, context_normalizer, config.context_fill_values,
                lambda m, x, c: m(x, c)["predictions"], bool(getattr(config, "starters_only", False)))

    return _predict_next_game_core(
        player_id, next_opponent, _load,
        df=df, defensive_target_stats=defensive_target_stats, player_bio=player_bio,
        schedule_context=schedule_context, player_age=player_age, injury_reports=injury_reports,
        injury_status=injury_status, archetypes=archetypes, defense_vs_archetype=defense_vs_archetype,
        position_group=position_group, device=device,
    )


if __name__ == "__main__":
    train_all_position_models_hf()

    # Example: after training, predict a specific player's next game against
    # a specific opponent.
    # print(predict_next_game_hf(player_id="00-0039732", next_opponent="LV"))  # Bo Nix vs. the Raiders

    # Example: push a trained model to the Hub (needs `hf auth login` first).
    # model = load_model_hf("QB")
    # model.push_to_hub("yourname/nfl-qb-transformer")
