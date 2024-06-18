# Unit 5.5. Organizing Your Data Loaders with Data Modules

import lightning as L
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import pandas as pd
import torch
from shared_utilities import LightningModel, MNISTDataModule, PyTorchMLP
from watermark import watermark

from lightning.pytorch.callbacks import Callback

class EvalMetricDiffCallback(Callback):
    def on_validation_end(self, trainer, lightning_model):
        train_acc = lightning_model.train_acc.compute()
        val_acc = lightning_model.val_acc.compute()
        diff = train_acc - val_acc
        print(f"Train-Validation accuracy difference: {diff*100:.2f}%")
        print("")

if __name__ == "__main__":

    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)

    dm = MNISTDataModule()
    callbacks = [EvalMetricDiffCallback()]

    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10, accelerator="cpu", devices=1, deterministic=True,
        logger=CSVLogger(save_dir="logs/", name="my-model"), callbacks=callbacks
    )
    trainer.fit(model=lightning_model, datamodule=dm)

    train_acc = trainer.test(dataloaders=dm.train_dataloader())[0]["test_acc"]
    val_acc = trainer.test(dataloaders=dm.val_dataloader())[0]["test_acc"]
    test_acc = trainer.test(datamodule=dm)[0]["test_acc"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )

# Plotting the logs
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

aggreg_metrics = []
agg_col = "epoch"
for i, dfg in metrics.groupby(agg_col):
    agg = dict(dfg.mean())
    agg[agg_col] = i
    aggreg_metrics.append(agg)

df_metrics = pd.DataFrame(aggreg_metrics)
df_metrics[["train_loss", "val_loss"]].plot(
    grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
)
df_metrics[["train_acc", "val_acc"]].plot(
    grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
)

plt.show()