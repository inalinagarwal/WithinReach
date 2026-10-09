"""Train sim-only, human-only or human-pretrained dynamics with identical sim budgets."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .common import read_records, save_json
from .model import Dynamics, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim", default="data/processed/sim.jsonl")
    p.add_argument("--human", default="data/processed/human.jsonl")
    p.add_argument("--mode", choices=["sim", "human", "transfer"], default="sim")
    p.add_argument("--sim-episodes", type=int, default=30)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--pretrain-epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/sim.pt")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sim = read_records(args.sim) if args.mode != "human" else []
    human = read_records(args.human) if args.mode != "sim" else []
    if any(r["source"] != "sim" for r in sim) or any(r["source"] != "human" for r in human):
        raise ValueError("The --sim and --human files must contain their respective data sources")
    for records in (sim, human):
        assignments = {}
        for record in records:
            episode, split = record["episode"], record["split"]
            if episode in assignments and assignments[episode] != split:
                raise ValueError(f"Episode leaks across train/val/test: {episode}")
            assignments[episode] = split
    sim_train = [r for r in sim if r["split"] == "train" and r["upright"]]
    episodes = sorted(set(r["episode"] for r in sim_train))
    rng.shuffle(episodes)
    if args.sim_episodes < 1 or args.epochs < 1 or args.pretrain_epochs < 1:
        p.error("Episode and epoch counts must be positive")
    chosen = set(episodes[:args.sim_episodes])
    sim_train = [r for r in sim_train if r["episode"] in chosen]
    human_train = [r for r in human if r["split"] == "train" and r["upright"]]
    stages = []
    if args.mode in ("human", "transfer"):
        stages.append(("human_pretraining", human_train, args.pretrain_epochs))
    if args.mode in ("sim", "transfer"):
        stages.append(("sim_adaptation", sim_train, args.epochs))
    if any(not records for _, records, _ in stages):
        raise ValueError("A training stage has no upright training examples. Check splits/data.")
    # Normalization uses only this run's training examples, never val/test outcomes.
    X, Y = arrays([r for _, records, _ in stages for r in records])
    xmean, xstd = X.mean(0), np.maximum(X.std(0), 1e-3)
    ymean, ystd = Y.mean(0), np.maximum(Y.std(0), [0.003, 0.003, 0.05]).astype(np.float32)
    model = Dynamics()
    history = []
    for name, records, epochs in stages:
        # Reset optimizer at adaptation; sim/transfer use the same update budget.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        X, Y = arrays(records)
        xt = torch.from_numpy((X - xmean) / xstd)
        yt = torch.from_numpy((Y - ymean) / ystd)
        for epoch in range(epochs):
            order = torch.randperm(len(records))
            losses = []
            for indices in order.split(64):
                loss = torch.mean((model(xt[indices]) - yt[indices]) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % 25 == 0 or epoch == epochs - 1:
                item = dict(stage=name, epoch=epoch + 1, normalized_mse=float(np.mean(losses)))
                history.append(item)
                print(item, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), xmean=torch.from_numpy(xmean),
                    xstd=torch.from_numpy(xstd), ymean=torch.from_numpy(ymean),
                    ystd=torch.from_numpy(ystd)), args.out)
    save_json(str(args.out) + ".metadata.json", dict(args=vars(args),
              sim_training_episodes=sorted(chosen), stages={n: len(r) for n, r, _ in stages},
              history=history, scope="Predict upright transitions only; no learned tip classifier."))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
