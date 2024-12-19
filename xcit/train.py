import models.xcit.XCiT as xcit_models
from dataset_utils.animal_utils.animals import get_animal_data_loader

from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch.nn.functional as F
import torch



class XCiTModel(pl.LightningModule):
    def __init__(self, num_classes=10, model_name="xcit_small_24_p16"):
        super().__init__()

        assert model_name in xcit_models.__dict__.keys(), f"Model {model_name} not found in xcit_models"
        self.model = xcit_models.__dict__[model_name](pretrained=False, num_classes=num_classes)
        print(f"Model {model_name} loaded")

        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        cls = self(x)
        loss = F.cross_entropy(cls, y)

        # 计算训练准确率
        correct = (cls.argmax(dim=-1) == y).sum().item()
        total = y.size(0)
        accuracy = correct / total

        # 记录训练损失和训练准确率
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=False, prog_bar=True)

        return {'loss': loss, 'train_acc': accuracy, 'num_samples': total}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        cls = self(x)
        loss = F.cross_entropy(cls, y)

        # 计算验证准确率
        correct = (cls.argmax(dim=-1) == y).sum().item()
        total = y.size(0)
        accuracy = correct / total

        # 记录验证损失和验证准确率
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=False, prog_bar=True)

        return {'loss': loss, 'val_acc': accuracy, 'num_samples': total}

    def training_epoch_end(self, outputs):
        # 计算训练的总准确率和总样本数
        total_correct = sum([x['train_acc'] * x['num_samples'] for x in outputs])
        total_samples = sum([x['num_samples'] for x in outputs])
        avg_train_acc = total_correct / total_samples

        # 计算训练损失的平均值
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # 输出训练的平均准确率和平均损失
        self.log('avg_train_acc', avg_train_acc, on_epoch=True, prog_bar=True)
        self.log('avg_train_loss', avg_train_loss, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs):
        # 计算验证的总准确率和总样本数
        total_correct = sum([x['val_acc'] * x['num_samples'] for x in outputs])
        total_samples = sum([x['num_samples'] for x in outputs])
        avg_val_acc = total_correct / total_samples

        # 计算验证损失的平均值
        avg_val_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # 输出验证的平均准确率和平均损失
        self.log('avg_val_acc', avg_val_acc, on_epoch=True, prog_bar=True)
        self.log('avg_val_loss', avg_val_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer





if __name__ == "__main__":
    trainer = Trainer(precision=16, gpus=1, max_epochs=10)
    train_loader, val_loader, num_classes = get_animal_data_loader(root_dir=r'D:\pyproject\representation_learning_models\dataset_utils\animals',
                                                                  batch_size=64,
                                                                 image_size=224)
    model = XCiTModel(num_classes=num_classes, model_name="xcit_small_24_p16")

    trainer.fit(model, train_loader, val_loader)



