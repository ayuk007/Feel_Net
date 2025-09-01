import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        class_names,
        save_dir="checkpoints",
        log_dir="runs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.class_names = class_names
        self.save_dir = save_dir
        self.log_dir = log_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}"):
            text_input = batch["text_input"].to(self.device)
            text_input_mask = batch["text_input_mask"].to(self.device)
            sentiment_input = batch["sentiment_input"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(text_input, text_input_mask)
            loss = self.loss_fn(outputs, sentiment_input.squeeze(1))
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += sentiment_input.size(0)

            current_correct = (predicted == sentiment_input.view(-1)).sum().item()
            correct += current_correct

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/Train", accuracy, epoch)
        return avg_loss, accuracy

    def validate_epoch(self, epoch, max_batch: int = 10):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Val Epoch {epoch+1}")):
                if batch_idx >= max_batch:
                    break
                text_input = batch["text_input"].to(self.device)
                text_input_mask = batch["text_input_mask"].to(self.device)
                sentiment_input = batch["sentiment_input"].to(self.device)

                outputs = self.model(text_input, text_input_mask)
                loss = self.loss_fn(outputs, sentiment_input.squeeze(1))
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += sentiment_input.size(0)
                correct += (predicted == sentiment_input.view(-1)).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(sentiment_input.squeeze(1).cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        # Classification report dict for logging
        report = classification_report(all_targets, all_preds, target_names=self.class_names, output_dict=True)
        self.writer.add_text("Classification_Report", classification_report(all_targets, all_preds, target_names=self.class_names), epoch)
        for cls in self.class_names:
            self.writer.add_scalar(f"F1_Score/{cls}", report[cls]["f1-score"], epoch)
            self.writer.add_scalar(f"Precision/{cls}", report[cls]["precision"], epoch)
            self.writer.add_scalar(f"Recall/{cls}", report[cls]["recall"], epoch)

        self.writer.add_scalar("Loss/Validation", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/Validation", accuracy, epoch)

        return avg_loss, accuracy, report

    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_dir, f"best_model_epoch_{epoch+1}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model checkpoint saved at {path}")

    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_report = self.validate_epoch(epoch)

            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Val   Accuracy: {val_acc:.2f}%")

            # if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.save_checkpoint(epoch)

        self.writer.close()
        print("Training complete.")