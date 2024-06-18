import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from dataset import CarDataset
from trainer_mps import train_step, val_step

torch.backends.cudnn.benchmark = True


def main(args, trainsforms=None):
    df = pd.read_csv('../../data/intro-dl-2024/train.csv')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=2024, stratify=df['label'])

    train_dataset = CarDataset(df_filename=train_df,
                               phase=args.phase,
                               img_size=args.img_size,
                               transforms=trainsforms)

    val_dataset = CarDataset(df_filename=test_df,
                             phase=args.phase,
                             img_size=args.img_size)

    model = args.model
    model = model.to(args.device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    scaler = torch.cuda.amp.GradScaler()

    device = args.device
    num_epoch = 100

    model = model.to(device)
    if device != "mps":
        model = torch.compile(model)
    best_val_acc = 0
    log_path = args.model_path[:-4]
    for epoch in range(1, 1 + num_epoch):
        train_step(train_loader, model, args.criterion, scaler, args.optimizer, device, epoch, log_path)

        val_accuracy = val_step(val_loader, model, args.criterion, device, log_path)
        args.lr_scheduler.step(val_accuracy)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
        torch.save(model.state_dict(), args.model_path)
