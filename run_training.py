import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import transformers
import wandb
import os

from dataset.dcase24 import get_training_set, get_test_set, get_eval_set
from helpers.init import worker_init_fn
from models.basic import get_model, get_model_v
from helpers import nessi


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.single_layer:
            self.model = get_model(
                J1=config.J1,
                J2=config.J2,
                m=config.m,
                n=config.n,
                alpha=config.alpha,
                beta=config.beta,
                mixstyle_p = config.mixstyle_p,
                mixstyle_alpha = config.mixstyle_alpha,
                skip_lp=config.skip_lp,
            )            
        self.model = get_model_v(
            J1=config.J1,
            alpha=config.alpha,
            beta=config.beta,
            model=config.model,
        )
    
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.example_input_array = torch.zeros(1, 1, 44100)

    def forward(self, x):
        x = self.model(x)
        return x


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr,
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]
    

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, files, labels, devices, cities = batch
        y_hat = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss.detach().cpu())
        return loss


    def on_train_epoch_end(self):
        pass


    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch

        y_hat = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)


    def on_validation_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'val' for logging
        self.log_dict({"val/" + k: logs[k] for k in logs})
        self.validation_step_outputs.clear()



def train(config):
    os.makedirs(config.sav_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'24 Task 1.",
        tags=["DCASE24"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name,
        save_dir=config.sav_dir,
        offline=True,
    )

    # train dataloader
    assert config.subset in {100, 50, 25, 10, 5}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."
    roll_samples = config.roll_sec * 44100
    dataset = get_training_set(
        split=config.subset,
        roll=roll_samples,
        dir_prob=config.dir_prob,
    )
    train_dl = DataLoader(dataset,
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    test_dl = DataLoader(dataset=get_test_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config)

    # get model complexity from nessi and log results to wandb
    shape = next(iter(test_dl))[0][0].unsqueeze(0).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    nessi.validate(macs, params)
    # log MACs and number of parameters for our model
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params
    wandb_logger.watch(pl_module, log_freq=50)

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    if config.fast_dev_run:
        trainer = pl.Trainer(
             accelerator="auto",
             devices=1,
             precision=config.precision,
             fast_dev_run=True,
             num_sanity_val_steps=2,
             profiler="simple",
             limit_train_batches=1,
        )
    else:   
        trainer = pl.Trainer(max_epochs=config.n_epochs,
                            logger=wandb_logger,
                            accelerator='gpu',
                            devices=1,
                            precision=config.precision,
                            gradient_clip_val=0.5,
                            callbacks=[
                                pl.callbacks.ModelCheckpoint(save_last=True),
                                ModelSummary(max_depth=-1)
                            ])
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, test_dl)

    # final test step
    # here: use the validation split
    wandb.finish()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 24 argument parser')

    # general
    parser.add_argument('--project_name', type=str, default="DCASE24_Task1")
    parser.add_argument('--experiment_name', type=str, default="Baseline")
    parser.add_argument('--num_workers', type=int, default=8)  # number of workers for dataloaders
    parser.add_argument('--precision', type=str, default="32")

    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default=None)  # for loading trained model, corresponds to wandb id

    # dataset
    # subset in {100, 50, 25, 10, 5}
    parser.add_argument('--subset', type=int, default=100)

    # model
    parser.add_argument('--alpha', type=int, default=33)
    parser.add_argument('--beta', type=int, default=10)
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--J1', type=int, default=8)
    parser.add_argument('--J2', type=int, default=4)
    parser.add_argument('--skip_lp', type=bool, default=False)
    parser.add_argument('--single_layer', type=bool, default=True)
    parser.add_argument('--model', type=str, default="conv1d")

    # augmentation
    parser.add_argument('--mixstyle_p', type=float, default=0.5)  # mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.2)
    parser.add_argument('--roll_sec', type=int, default=0)  # roll waveform over time
    parser.add_argument('--dir_prob', type=float, default=0.4) # prob. to apply device impulse response augmentation


    # training
    parser.add_argument('--sav_dir', type=str, default="./log")
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--fast_dev_run', type=bool, default=False) 

    # peak learning rate (in cosinge schedule)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--warmup_steps', type=int, default=2000)


    args = parser.parse_args()
    train(args)
