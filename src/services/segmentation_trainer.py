import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm


class SegmentationTrainer:
    def __init__(
        self,
        data_module,
        num_classes=5,
        lr=1e-4,
        class_weights=None,
        use_pretrained=True,
        device=None,
    ):
        self.data_module = data_module
        self.num_classes = num_classes
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model with optional pretrained weights
        if use_pretrained:
            print("Using pretrained ResNet50 backbone (ImageNet weights)")
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

            # Load model with pretrained weights (21 classes for COCO)
            self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

            # Replace the classifier head for our num_classes
            # The classifier is in model.classifier[4]
            self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            # Also update the auxiliary classifier if it exists
            if hasattr(self.model, "aux_classifier"):
                self.model.aux_classifier[4] = nn.Conv2d(
                    256, num_classes, kernel_size=1
                )
        else:
            print("Training from scratch (no pretrained weights)")
            self.model = deeplabv3_resnet50(weights=None, num_classes=num_classes)

        self.model = self.model.to(self.device)

        # Loss function with optional class weights
        if class_weights is not None:
            print(f"Using class weights: {class_weights.tolist()}")
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        else:
            print("No class weights specified (uniform weighting)")
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        num_valid_batches = 0

        for images, masks in tqdm(self.data_module.train_loader, desc="Training"):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)["out"]
            loss = self.criterion(outputs, masks)

            # Skip batches where loss is NaN (happens when all pixels are ignore_index)
            if not torch.isnan(loss):
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * images.size(0)
                num_valid_batches += 1

        # Avoid division by zero if all batches were invalid
        if num_valid_batches == 0:
            return float("nan")

        avg_loss = train_loss / len(self.data_module.train_dataset)
        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        num_valid_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(self.data_module.val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)["out"]
                loss = self.criterion(outputs, masks)

                # Skip batches where loss is NaN (happens when all pixels are ignore_index)
                if not torch.isnan(loss):
                    val_loss += loss.item() * images.size(0)
                    num_valid_batches += 1

        # Avoid division by zero if all batches were invalid
        if num_valid_batches == 0:
            return float("nan")

        avg_loss = val_loss / len(self.data_module.val_dataset)
        return avg_loss

    def fit(self, epochs=5):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

    def predict(self, images):
        """Predict segmentation masks for a batch of images"""
        self.model.eval()
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.model(images)["out"]
            preds = torch.argmax(outputs, dim=1)
        return preds.cpu()
