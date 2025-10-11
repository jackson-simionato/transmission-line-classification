from torch.utils.data import DataLoader
from torchvision import transforms
from src.services.dataset_service import SegmentationDataset
from src.config.settings import get_config


class DataLoaderService:
    def __init__(self, images_dir, masks_dir, batch_size=4, num_workers=2):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = get_config()

        # Define transforms (you can add augmentations here)
        self.train_transform = transforms.Compose([
            # e.g., RandomHorizontalFlip(), RandomRotation(10)
        ])
        self.val_transform = None
        self.test_transform = None

        # Placeholders for datasets and dataloaders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self):
        """Create datasets and dataloaders"""
        self.train_dataset = SegmentationDataset(
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            split_ids=self.config.TRAIN_DATASET_IDS,
            transform=self.train_transform
        )

        self.val_dataset = SegmentationDataset(
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            split_ids=self.config.VAL_DATASET_IDS,
            transform=self.val_transform
        )

        self.test_dataset = SegmentationDataset(
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            split_ids=self.config.TEST_DATASET_IDS,
            transform=self.test_transform
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
