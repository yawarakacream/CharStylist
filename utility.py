import os

import torchvision


def char2code(char):
    return format(ord(char), '#06x')

# def code2char(code):
#     chr(int(code, base=16))

def pathstr(t):
    return os.path.expanduser(t)

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    image = torchvision.transforms.ToPILImage()(grid)
    image.save(path)
    return image

def save_single_image(image, path):
    image = torchvision.transforms.ToPILImage()(image)
    image.save(path)
    return image
