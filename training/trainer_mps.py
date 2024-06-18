import torch
from utils import AverageMeter
from datetime import datetime

start_time = None
torch.backends.cudnn.benchmark = True


def get_log_string(start_time, s):
    time_from_start = str(datetime.now() - start_time)
    return '\r' + time_from_start + ' ' + s


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def val_step(val_loader, model, criterion, device, log_path):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for img, targets in val_loader:
            targets = targets.to(device)
            inputs = img.to(device)

            if device != "mps":
                with torch.autocast(device_type=device, dtype=torch.float16):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs, 1)
            prec1 = predicted.eq(targets.data).cpu().sum()

            losses.update(to_python_float(val_loss.data), inputs.size(0))
            top1.update(to_python_float(prec1) / inputs.size(0), inputs.size(0))

        print('')
        log_string = get_log_string(start_time,
                                    '*Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))
        print(log_string)
        with open(log_path, "a") as log_file:
            log_file.write(log_string)
    return top1.avg


def train_step(train_loader, model, criterion, scaler, optimizer, device, epoch, log_path):
    global start_time
    start_time = datetime.now()
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_idx = 0
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if device != "mps":
            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs = model(inputs)
        else:
            outputs = model(inputs)

        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        prec1 = predicted.eq(targets.data).cpu().sum()

        losses.update(to_python_float(loss.data), inputs.size(0))
        top1.update(to_python_float(prec1) / inputs.size(0), inputs.size(0))

        optimizer.zero_grad()
        # loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

        batch_idx += 1
        log_string = get_log_string(start_time, 'Epoch: [{0}][{1}/{2}]\t'
                                                'Loss {loss.avg:.7f}\t'
                                                'Prec@1 {top1.avg:.3f}'.format(epoch, batch_idx, len(train_loader),
                                                                               loss=losses, top1=top1))

        print(log_string)
        with open(log_path, "a") as log_file:
            log_file.write(log_string)
