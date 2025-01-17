import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms


def show_tensor_image(tensor_img, image_processor=None):
    """
    Display a tensor image
    Args:
        tensor_img: Image in tensor format (C x H x W)
    """
    if image_processor: 
        mean = image_processor.image_mean  
        std = image_processor.image_std
        denormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )
        tensor_img = denormalize(tensor_img)

    img_np = tensor_img.permute(1, 2, 0).numpy()

    # if img_np.max() <= 1:
    #     img_np = (img_np * 255).astype("uint8")

    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    plt.axis("off")
    plt.show()


def show_batch(dataloader, num_images=16):
    """
    Display a batch of images from the dataloader
    """
    batch = next(iter(dataloader))
    images = batch["image"][:num_images]
    labels = batch["labels"][:num_images]

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for i, (img, label) in enumerate(zip(images, labels)):
        img_np = img.permute(1, 2, 0).numpy()
        if img_np.max() <= 1:
            img_np = (img_np * 255).astype("uint8")

        ax = axes[i // 4, i % 4]
        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(f"Emotions: {label.nonzero().squeeze().tolist()}")

    plt.tight_layout()
    plt.show()


def split_dataset(dataset, train_size, val_size, random=False):
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    if random:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size]
        )
    else:
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size + val_size]
        test_dataset = dataset[train_size + val_size:]
    return train_dataset, val_dataset, test_dataset


augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),               
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(degrees=15, scale=(0.9, 1.1), shear=5),
])
