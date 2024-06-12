# 88-key 피아노 연주를 입력 받아 10-key 버튼 입력으로 맵핑하고, 이를 다시 88-key 피아노 연주로 되돌리는 오토 인코더를 만들고 학습하는 코드
# Piano Genie 코드를 참고/수정 하였음 (출처: https://github.com/chrisdonahue/music-cocreation-tutorial/tree/main)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import random
import numpy as np
import json

PIANO_LOWEST_KEY_MIDI_PITCH = 21
PIANO_NUM_KEYS = 88
MASTER_TOKEN = 10
SOS = PIANO_NUM_KEYS

CFG = {
    "seed": 0,
    # Number of buttons in interface
    "num_buttons": 10,
    # Onset delta times will be clipped to this maximum
    "data_delta_time_max": 2.0,
    # RNN dimensionality
    "model_rnn_dim": 1024,
    # RNN num layers
    "model_rnn_num_layers": 2,
    # Training hyperparameters
    "batch_size": 32,
    "seq_len": 128,
    "loss_margin_multiplier": 1.0,
    "loss_contour_multiplier": 1.0,
    "summarize_frequency": 128,
    "eval_frequency": 128,
    "max_num_steps": 15000,
    "max_lr": 5.0e-4,
    "min_lr": 5.0e-5,
    "lr_decay_steps": 15000,
    "lr_warmup_steps": 200,
}


# t=0~i-1 시점 까지의 피아노 연주(key, time), t=i 시점의 버튼 입력(button), 곡의 조성(mode)을 입력 받아서 t=i 시점의 피아노 음을 만들어 내는 Decoder
# 학습 과정에서는 피아노 연주로 실제 원본 연주를, 버튼 입력으로는 인코더의 아웃풋을 입력 데이터로 사용
# 피아노 연주는 피아노 음(key)와 음 사이의 간격(time)으로 표현
class GhostPianistDecoder(nn.Module):
    def __init__(self, rnn_dim=128, rnn_num_layers=2):
        super().__init__()
        self.rnn_dim = rnn_dim
        self.rnn_num_layers = rnn_num_layers
        self.input = nn.Linear(PIANO_NUM_KEYS + 3 + 24, rnn_dim)
        self.lstm = nn.LSTM(
            rnn_dim,
            rnn_dim,
            rnn_num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.output = nn.Linear(rnn_dim, 88)


    def forward(self, key, time, button, mode):
        # 이전 피아노 연주를 입력으로 받아 현재 피아노 음을 예측하게 만들기 위해 <S> 토큰을 시퀀스의 맨 앞에 추가
        new_key = torch.cat([torch.full_like(key[:, :1], SOS), key[:, :-1]], dim=1)
        new_time = torch.cat([torch.full_like(time[:, :1], 1e8), key[:, :-1]], dim=1)

        # Encode input
        inputs = [
            F.one_hot(new_key, PIANO_NUM_KEYS + 1),
            new_time.unsqueeze(dim=2),
            button.unsqueeze(dim=2),
            F.one_hot(mode, 24),
        ]
        x = torch.cat(inputs, dim=2)

        # Project encoded inputs
        x = self.input(x)

        # Run
        x, h_N = self.lstm(x, None)

        # Compute logits
        hat_k = self.output(x)

        return hat_k


# 88-key 피아노 연주를 입력으로 받아 10-key 버튼 입력으로 맵핑하는 encoder
class GhostPianistEncoder(nn.Module):
    def __init__(self, rnn_dim=128, rnn_num_layers=2):
        super().__init__()
        self.rnn_dim = rnn_dim
        self.rnn_num_layers = rnn_num_layers
        self.input = nn.Linear(PIANO_NUM_KEYS + 1, rnn_dim)
        self.lstm = nn.LSTM(
            rnn_dim,
            rnn_dim,
            rnn_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(rnn_dim * 2, 1)

    def forward(self, key, time):
        # Encode input
        inputs = [
            F.one_hot(key, PIANO_NUM_KEYS),
            time.unsqueeze(dim=2),
        ]
        x = torch.cat(inputs, dim=2)

        # Project encoded inputs
        x = self.input(x)

        # Run
        x, _ = self.lstm(x, None)

        # Project LSTM output
        x = self.output(x)

        return x[:, :, 0]


# 실수로 표현된 버튼 입력을 정수로 양자화 시키는 모듈
class IntegerQuantizer(nn.Module):
    def __init__(self, num_bins):
        super().__init__()
        self.num_bins = num_bins

    def real_to_discrete(self, x, eps=1e-6):
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        x *= self.num_bins - 1
        x = (torch.round(x) + eps).long()
        return x

    def discrete_to_real(self, x):
        x = x.float()
        x /= self.num_bins - 1
        x = (x * 2) - 1
        return x

    def forward(self, x):
        # Quantize and compute delta (used for straight-through estimator)
        with torch.no_grad():
            x_disc = self.real_to_discrete(x)
            x_quant = self.discrete_to_real(x_disc)
            x_quant_delta = x_quant - x

        x = x + x_quant_delta

        return x


# 88-key 피아노 입력을 10-key 버튼 입력으로 맵핑하고, 이를 다시 88-key 피아노 입력으로 되돌리는 오토 인코더
# 실제 연주 과정에서는 디코더만을 사용하고 사용자의 버튼 입력을 input으로 사용
class GhostPianistAutoencoder(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.enc = GhostPianistEncoder(
            rnn_dim=CFG["model_rnn_dim"],
            rnn_num_layers=CFG["model_rnn_num_layers"],
        )

        self.quant = IntegerQuantizer(CFG["num_buttons"])

        self.dec = GhostPianistDecoder(
            rnn_dim=CFG["model_rnn_dim"],
            rnn_num_layers=CFG["model_rnn_num_layers"],
        )

    def forward(self, key, time, mode):
        # 인코더가 88-key 피아노 연주를 입력 받아 10-key 버튼 입력으로 맵핑하는 과정, 학습 과정에서 버튼 입력은 실수로 표현
        e = self.enc(key, time)
        button = self.quant(e)

        # 화음의 일부를 확률적으로 master token으로 바꾸어서 디코더가 해당 위치의 버튼 정보가 없어도 대응되는 피아노 음을 예측하게 만들어 줌
        # 실제 연주 과정에서 사용자가 누른 버튼의 수 보다 많은 피아노 음을 연주 하고자 할 때 master token을 추가하여 추가적인 피아노 음을 만들 수 있게 함
        mask = (time < 0.005).float()
        random_mask = torch.bernoulli(0.5 * torch.ones_like(time))
        final_mask = mask * random_mask
        button[final_mask > 0.5] = MASTER_TOKEN

        # 디코더가 이전 피아노 연주, 현재 버튼 입력, 곡의 조성을 입력 받아 현재 연주될 피아노 음을 예측 하는 과정
        hat_k = self.dec(key, time, button, mode)
        return hat_k, e


# 학습에 필요한 배치를 만드는 함수
def performances_to_batch(performances, train=True):
    batch_k = []
    batch_t = []
    batch_m = []
    for p in performances:
        assert len(p) >= CFG["seq_len"]
        if train:
            subsample_offset = random.randrange(0, len(p) - CFG["seq_len"])
        else:
            subsample_offset = 0
        subsample = p[subsample_offset: subsample_offset + CFG["seq_len"]]
        assert len(subsample) == CFG["seq_len"]

        # Key features
        batch_k.append([n[2] for n in subsample])

        # Mode features
        batch_m.append([n[4] for n in subsample])

        # Onset features
        # NOTE: For stability, we pass delta time
        t = np.diff([n[0] for n in subsample])
        t = np.concatenate([[1e8], t])
        t = np.clip(t, 0, CFG["data_delta_time_max"])
        batch_t.append(t)

    return torch.tensor(batch_k).long(), torch.tensor(batch_t).float(), torch.tensor(batch_m).long()


if __name__ == "__main__":
    # Init
    run_dir = pathlib.Path("ghost_pianist")
    run_dir.mkdir(exist_ok=True)

    # Set seed
    if CFG["seed"] is not None:
        random.seed(CFG["seed"])
        np.random.seed(CFG["seed"])
        torch.manual_seed(CFG["seed"])
        torch.cuda.manual_seed_all(CFG["seed"])

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GhostPianistAutoencoder(CFG)
    model.train()
    model.to(device)
    print("-" * 80)
    for n, p in model.named_parameters():
        print(f"{n}, {p.shape}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["max_lr"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, CFG["lr_decay_steps"], eta_min=CFG["min_lr"]
    )

    # Load dataset
    dataset_path = './Dataset/dataset.json'
    with open(dataset_path, "r") as st_json:
        DATASET = json.load(st_json)

    # Train
    step = 0
    best_eval_loss = float("inf")

    while CFG["max_num_steps"] is None or step < CFG["max_num_steps"]:
        if step % CFG["eval_frequency"] == 0:
            model.eval()

            with torch.no_grad():
                eval_losses_recons = []
                eval_violates_contour = []
                for i in range(0, len(DATASET["validation"]), CFG["batch_size"]):
                    eval_batch = performances_to_batch(
                        DATASET["validation"][i: i + CFG["batch_size"]],
                        train=False,
                    )
                    eval_k, eval_t, eval_m = tuple(t.to(device) for t in eval_batch)
                    eval_hat_k, eval_e = model(eval_k, eval_t, eval_m)
                    eval_b = model.quant.real_to_discrete(eval_e)

                    # 오토 인코더의 입력과 아웃풋이 얼마나 유사한지 나타내는 로스
                    eval_loss_recons = F.cross_entropy(
                        eval_hat_k.view(-1, PIANO_NUM_KEYS),
                        eval_k.view(-1),
                        reduction="none",
                    )

                    # 원본 피아노 연주의 피아노 음의 오르내림과 인코더가 맵핑한 버튼 입력의 오르내림이 얼마나 유사한지 나타내는 로스
                    eval_violates = torch.logical_not(
                        torch.sign(torch.diff(eval_k, dim=1))
                        == torch.sign(torch.diff(eval_b, dim=1)),
                    ).float()

                    eval_violates_contour.extend(eval_violates.cpu().numpy().tolist())
                    eval_losses_recons.extend(eval_loss_recons.cpu().numpy().tolist())

                print('-' * 10)
                print(eval_k[0].tolist())
                print(torch.max(eval_hat_k, dim=2).indices[0].tolist())
                print(eval_b[0].tolist())
                print(eval_t[0].tolist())

                eval_loss_recons = np.mean(eval_losses_recons)
                # 베스트 모델을 저장
                if eval_loss_recons < best_eval_loss:
                    torch.save(model.state_dict(), pathlib.Path(run_dir, "model.pt"))
                    best_eval_loss = eval_loss_recons

            eval_metrics = {
                "eval_loss_recons": eval_loss_recons,
                "eval_contour_violation_ratio": np.mean(eval_violates_contour),
            }

            print(step, "eval", eval_metrics)

            model.train()

        # Create minibatch
        batch = performances_to_batch(
            random.sample(DATASET["train"], CFG["batch_size"]), train=True
        )
        k, t, m = tuple(t.to(device) for t in batch)

        # Run model
        optimizer.zero_grad()
        k_hat, e = model(k, t, m)

        # Compute losses and update params
        # 오토 인코더의 입력과 아웃풋이 얼마나 유사한지 검사하는 로스
        loss_recons = F.cross_entropy(k_hat.view(-1, PIANO_NUM_KEYS), k.view(-1))
        # 인코더가 맵핑한 버튼 입력의 범위가 [-1,1]을 넘어가지 않게 만드는 로스
        loss_margin = torch.square(
            torch.maximum(torch.abs(e) - 1, torch.zeros_like(e))
        ).mean()
        # 원본 피아노 연주의 피아노 음의 오르내림과 인코더가 맵핑한 버튼 입력의 오르내림이 얼마나 유사한지 검사하는 로스
        loss_contour = torch.square(
            torch.maximum(
                1 - torch.diff(k, dim=1) * torch.diff(e, dim=1),
                torch.zeros_like(e[:, 1:]),
            )
        ).mean()
        loss = torch.zeros_like(loss_recons)
        loss += loss_recons
        if CFG["loss_margin_multiplier"] > 0:
            loss += CFG["loss_margin_multiplier"] * loss_margin
        if CFG["loss_contour_multiplier"] > 0:
            loss += CFG["loss_contour_multiplier"] * loss_contour

        # anneal learning rate
        if step < CFG["lr_warmup_steps"]:
            curr_lr = CFG["max_lr"] * step / CFG["lr_warmup_steps"]
            optimizer.param_groups[0]['lr'] = curr_lr
        else:
            scheduler.step(step - CFG["lr_warmup_steps"])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        step += 1

        if step % CFG["summarize_frequency"] == 0:
            metrics = {
                "loss_recons": loss_recons.item(),
                "loss_margin": loss_margin.item(),
                "loss_contour": loss_contour.item(),
                "loss": loss.item(),
            }

            print(step, "train", metrics)
