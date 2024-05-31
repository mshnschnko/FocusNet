from torchvision import transforms

class ConditionalRandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.random_crop = transforms.RandomCrop((crop_size, crop_size))

    def __call__(self, img):
        if img.size[0] > self.crop_size and img.size[1] > self.crop_size:
            return self.random_crop(img)
        return img
