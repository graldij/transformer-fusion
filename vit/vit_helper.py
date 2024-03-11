import math
import torch
from transformers import ViTConfig, ViTForImageClassification
from transformers import Trainer, TrainingArguments
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from torchvision import transforms
from sklearn.metrics import accuracy_score
import numpy as np
from datasets import load_dataset

_val_transforms_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)),
])

# PUBLIC
def val_transforms_cifar10(examples):
    examples['pixel_values'] = [_val_transforms_cifar10(img.convert("RGB")) for img in examples['img']]
    return examples

def load_dataset_vit(dataset, seed = 0):
    if dataset=="cifar10":
        train_ds, test_ds = load_dataset("cifar10", split=['train', 'test'])
        splits = train_ds.train_test_split(test_size=0.1, seed=seed)
        train_ds = splits['train']
        val_ds = splits['test']
        # Transforms are done on the fly in a lazy way
        # Setting up the transforms on each dataset
        val_ds.set_transform(val_transforms_cifar10)
        test_ds.set_transform(val_transforms_cifar10)
    else:
        raise NotImplementedError
    return val_ds, test_ds


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


def get_new_model(dataset, patch_size, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size, hidden_dropout_prob, attention_probs_dropout_prob):
    # Definining the model from scratch
    if dataset == "cifar10":
        configs = ViTConfig(image_size=32,
                                patch_size = patch_size,
                                num_hidden_layers = num_hidden_layers,
                                num_attention_heads = num_attention_heads,
                                hidden_size = hidden_size,
                                intermediate_size = intermediate_size,
                                hidden_act = 'gelu',
                                hidden_dropout_prob = hidden_dropout_prob,
                                attention_probs_dropout_prob = attention_probs_dropout_prob)

        configs.num_labels = 10
        model = ViTForImageClassification(configs)
    else:
        raise NotImplementedError
    return model


def get_model(path):
    return ViTForImageClassification.from_pretrained(path)


def compute_tot_iters(ds, epoches, bs, grad_acc_steps):
    tot_iters = len(ds)*epoches//bs//grad_acc_steps
    return tot_iters


def get_cosine_lr_wup_rstr(opt, ds, epoches, bs, wup_ratio, num_cycles, grad_acc_steps):
    tot_iters = compute_tot_iters(ds, epoches, bs, grad_acc_steps)
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer = opt, 
        num_warmup_steps= int(wup_ratio*tot_iters),
        num_training_steps = tot_iters,
        num_cycles = num_cycles
    )
    return lr_scheduler


def get_train_args(training_name, wup_ratio, lr, train_bs, eval_bs, epochs, wd, n_workers, grad_acc_steps, label_smoothing, seed, train_ds_len, report_to="wandb"):
    steps_per_epoches = math.ceil(train_ds_len/(train_bs*grad_acc_steps))
    n_epochs_save = 10
    args = TrainingArguments(
        training_name,
        save_strategy="steps",
        save_steps=steps_per_epoches*n_epochs_save,
        evaluation_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=wup_ratio,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        weight_decay=wd,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
        dataloader_num_workers=n_workers,
        gradient_accumulation_steps=grad_acc_steps,
        report_to=report_to,
        label_smoothing_factor=label_smoothing,
        logging_steps=steps_per_epoches,
        seed=seed
    )
    return args


def evaluate_vit(model, test_ds):
    args = TrainingArguments(
        "eval_temp",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_eval_batch_size=128,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
        dataloader_num_workers=16,
    )

    trainer = Trainer(
        model,
        args,
        eval_dataset=test_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    outputs = trainer.predict(test_ds)

    return outputs.metrics['test_accuracy']