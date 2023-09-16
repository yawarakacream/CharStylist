import os
import argparse

from distutils.util import strtobool

from char_stylist import CharStylist
from utility import char2code, pathstr, save_images, save_single_image


def main(
    save_path,
    stable_diffusion_path,
    
    chars,
    writers,
    
    grid,
    
    device,
):
    char_stylist = CharStylist.load(
        save_path,
        stable_diffusion_path,
        
        device,
    )
    
    for char in chars:
        if char not in char_stylist.char2idx:
            raise Exception(f"unknown character: {char}")
    
    for writer in writers:
        if writer not in char_stylist.writer2idx:
            raise Exception(f"unknown writer: {writer}")
    
    # sampling
    for char in chars:
        images = char_stylist.sampling(char, writers)
        
        if grid:
            image_path = os.path.join(save_path, 'generated', f"{char2code(char)} writers={','.join(writers)}.jpg")
            sampled_ema = save_images(images, image_path)
        
        else:
            for writer, image in zip(writers, images):
                image_path = os.path.join(save_path, 'generated', f"{char2code(char)} writer={writer}.png")
                sampled_ema = save_single_image(image, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_path", type=pathstr, default=pathstr("./datadisk/save_path/test ETL4,ETL5"))
    parser.add_argument("--stable_diffusion_path", type=pathstr, default=pathstr("~/datadisk/stable-diffusion-v1-5"))
    
    parser.add_argument("--chars", type=str, nargs="*", default=["あ", "コ"])
    parser.add_argument("--writers", type=str, nargs="*", default=["ETL4_5001", "ETL5_6001"])
    parser.add_argument("--grid", type=strtobool, default=True)
    
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    # todo: validation
    
    main(
        args.save_path,
        args.stable_diffusion_path,
        
        args.chars,
        args.writers,
        
        args.grid,
        
        args.device,
    )
