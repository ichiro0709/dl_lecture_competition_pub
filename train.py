import torch
import hydra
from omegaconf import DictConfig
import time
from main import VQA_criterion


from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="base")
    
def train(model, dataloader, optimizer, criterion, device, args: DictConfig):
    model.train()
    set_seed(args.seed)
    scaler = torch.cuda.amp.GradScaler()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(image, question)
            loss = criterion(pred, mode_answer.squeeze())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

if __name__ == "__main__":
    train()
