import torch
from torch.utils.data import Dataset
from PIL import Image


class ArtEmotionDataset(Dataset):
    def __init__(self, dataframe, wikiart_dir, image_processor, augmentations=None):
        """
        Args:
            dataframe: The artemis_grouped DataFrame
            wikiart_dir: Path to WikiArt directory
            parent_dir: Parent directory path
            transform: Optional transform to be applied on images
        """
        self.dataframe = dataframe
        self.wikiart_dir = wikiart_dir
        self.image_processor = image_processor
        self.augmentations = augmentations

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ArtEmotionDataset(self.dataframe[idx], self.wikiart_dir, self.image_processor, self.augmentations)
        
        row = self.dataframe.iloc[idx]
        # Get image path
        art_style = row["art_style"]
        painting = row["painting"]
        image_path = f"{self.wikiart_dir}/{art_style}/{painting}.jpg"

        # Load and transform image
        image = Image.open(image_path).convert("RGB")

        if self.augmentations: 
            image = self.augmentations(image)
        
        image = self.image_processor(image, return_tensors="pt")
        image = image['pixel_values'].squeeze(0)

        labels = torch.tensor(row['binary_labels'], dtype=torch.float32)

        return {
            'image': image,
            'labels': labels,
            'style': art_style,
            'painting': painting,
        }
