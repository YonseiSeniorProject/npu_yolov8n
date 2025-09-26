from myyololib.loss import v8DetectionLoss
from myyololib.dataset import preprocess_dataset
from myyololib.evaluator import evaluate
import torch
import torch.optim as optim
import time

def train(model, train_loader, val_loader, coco_id2label, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Hyperparameters
    num_epochs = cfg.num_epochs
    lr = cfg.learning_rate
    momentum = cfg.momentum
    weight_decay = cfg.weight_decay
    batch_size = cfg.batch_size
    step_size = cfg.step_size
    gamma = cfg.gamma
    validation_frequency = cfg.validation_frequency
    print_info = cfg.print_info

    # Criterion
    criterion = v8DetectionLoss(tal_topk=10)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
        )

    # scheduler 
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=step_size, 
        gamma=gamma
        )

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        cnt_correct = 0
        cnt_total = 0
        start = time.time()
        epoch_start = time.time()

        for i, (images, annotations) in enumerate(train_loader): 
            input, batch = preprocess_dataset(images, annotations, coco_id2label, device) # preprocess dataset 

            optimizer.zero_grad()
            outputs = model(input)
            loss, loss_det = criterion(outputs, batch)
            total_loss = loss.sum()
            total_loss.backward()
            optimizer.step()

            total_loss = loss.sum()
            
            if (i % 128) == 127:
                print(i+1, "/", len(train_loader), "time:", f"{((time.time()-start)):.4f}", "sec loss:", loss_det, optimizer.param_groups[0]['lr'])
                start = time.time()
            
            running_loss += loss_det.cpu()
            cnt_total += 16

        lr_cache = optimizer.param_groups[0]['lr']
        scheduler.step()

        train_loss = running_loss / cnt_total * 16

        # Optional validation every `validation_frequency` epochs
        if (epoch + 1) % validation_frequency == 0:
            evaluate(model, val_loader, map_only=True)
        if print_info:
            print(f"Epoch: {epoch+1}/{num_epochs} \t| "
                f"Train Loss: {train_loss} \t| "
                f"Time: {time.time() - epoch_start:.2f}s"
                f"LR: {lr_cache:.6f} ")
            epoch_start = time.time()
