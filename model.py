from typing import *

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim


class ConvBnDropBlock(nn.Sequential):
    "Create sequence of convolutional, Activation, `BatchNorm` & Drouput layers."

    def __init__(
        self,
        in_chans,
        out_chans,
        kernel_size,
        stride=1,
        dilation=1,
        padding=None,
        bias=True,
        act_cls: Callable = nn.ReLU,
        p_drop=0.0,
        use_bn=True,
    ):

        if padding is None:
            # for same padding
            padding = (kernel_size - 1) // 2

        layers = []
        # Initialize the convolutional layer
        conv_layer = nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size,
            stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )

        if act_cls is not None:
            act_layer = act_cls(inplace=True)
        else:
            act_layer = nn.Identity()
        if use_bn:
            norm_layer = nn.BatchNorm2d(out_chans)
        else:
            norm_layer = nn.Identity()

        layers += [conv_layer, act_layer, norm_layer]

        if p_drop > 0.0:
            layers.append(nn.Dropout2d(p=p_drop))

        super(ConvBnDropBlock, self).__init__(*layers)


class xResBlock(nn.Module):
    "Creates a simple Residual Block for xResNet architecture"

    def __init__(self, in_chans, out_chans, kernel_size, stride=1, act_cls=nn.ReLU):
        super(xResBlock, self).__init__()
        self.idconv = None

        self.block1 = ConvBnDropBlock(
            in_chans=in_chans,
            out_chans=out_chans,
            kernel_size=kernel_size,
            stride=1,
            use_bn=True,
            act_cls=act_cls,
            bias=False,
        )

        # we apply the 1st change here,
        # moving the stride 2 to the second convolution and keeps a stride of 1 for the first layer .
        self.block2 = ConvBnDropBlock(
            in_chans=out_chans,
            out_chans=out_chans,
            kernel_size=kernel_size,
            stride=stride,
            use_bn=True,
            act_cls=None,
            bias=False,
        )

        self.act_cls = act_cls(inplace=True)

        if in_chans != out_chans or stride != 1:
            # the 3rd change proped above is applied here,
            # we replace with a 2x2 average-pooling layer of stride 2 followed by a 1x1 convolution layer
            pool_layer = nn.AvgPool2d(stride, ceil_mode=True)
            conv_layer = ConvBnDropBlock(
                in_chans=in_chans,
                out_chans=out_chans,
                kernel_size=1,
                padding=0,
                use_bn=True,
                act_cls=None,
                p_drop=0.0,
                bias=False,
                stride=1,
            )

            self.idconv = nn.Sequential(pool_layer, conv_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.block2.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)

        if self.idconv is not None:
            identity = self.idconv(x)

        out += identity
        return self.act_cls(out)


@torch.jit.script
def mish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Mish, self).__init__()

    def forward(self, xb):
        return mish(xb)


class xResModel(nn.Sequential):
    def __init__(self, num_outputs: int, act_cls=Mish):
        conv = nn.Sequential(
            ConvBnDropBlock(
                in_chans=3,
                out_chans=32,
                kernel_size=3,
                stride=2,
                act_cls=act_cls,
                bias=False,
                use_bn=False,
            ),
            ConvBnDropBlock(
                in_chans=32,
                out_chans=32,
                kernel_size=3,
                stride=1,
                act_cls=act_cls,
                bias=False,
                use_bn=False,
            ),
            ConvBnDropBlock(
                in_chans=32,
                out_chans=64,
                kernel_size=3,
                stride=1,
                act_cls=act_cls,
                bias=False,
                use_bn=False,
            ),
        )
        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        conv_stem = nn.Sequential(conv, pool)
        block1 = xResBlock(
            in_chans=64, out_chans=64, stride=1, kernel_size=3, act_cls=act_cls
        )
        block2 = xResBlock(
            in_chans=64, out_chans=128, stride=2, kernel_size=3, act_cls=act_cls
        )
        block3 = xResBlock(
            in_chans=128, out_chans=256, stride=2, kernel_size=3, act_cls=act_cls
        )
        pool_flatten = nn.Sequential(
            OrderedDict(pool=nn.AdaptiveAvgPool2d(1), flatten=nn.Flatten())
        )
        fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(256, num_outputs))
        layers = OrderedDict(
            stem=conv_stem,
            block1=block1,
            block2=block2,
            block3=block3,
            pool_flatten=pool_flatten,
            fc=fc,
        )
        super(xResModel, self).__init__(layers)


class ClassificationTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        wd: float = 0,
        criterion: nn.Module = nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.save_hyperparameters("lr", "wd")
        self.model = model
        self.criterion = criterion

    def forward(self, xb):
        "Same as nn.Module forward"
        return self.model(xb)

    def shared_step(self, batch, batch_idx, *args, **kwargs):
        """
        The common step shared between the training, validation & test steps
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = FM.accuracy(F.softmax(y_hat), y)
        metrics = {"accuracy": acc, "loss": loss}
        return metrics

    def training_step(self, batch, batch_idx, *args, **kwargs):
        metrics = self.shared_step(batch, batch_idx, *args, **kwargs)
        metrics = {"train_acc": metrics["accuracy"], "train_loss": metrics["loss"]}
        self.log_dict(metrics)
        return metrics["train_loss"]

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        metrics = self.shared_step(batch, batch_idx, *args, **kwargs)
        metrics = {"val_acc": metrics["accuracy"], "val_loss": metrics["loss"]}
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        metrics = self.shared_step(batch, batch_idx, *args, **kwargs)
        metrics = {"test_acc": metrics["accuracy"], "test_loss": metrics["loss"]}
        self.log_dict(metrics)

    def configure_optimizers(self):
        """
        define optimizers and LR schedulers for use in training.
        """
        # default Adam parameters from fast.ai
        opt = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(0.9, 0.99),
            eps=1e-05,
        )
        steps = len(self.train_dataloader())
        epochs = self.trainer.max_epochs

        scheduler = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.lr, epochs=epochs, steps_per_epoch=steps
        )
        return [opt], [dict(scheduler=scheduler, interval="step")]
