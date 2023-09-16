import os
import argparse

from tqdm import tqdm

from char_stylist import CharStylist
from utility import char2code, pathstr, save_images, save_single_image

import numpy as np
import torch


def main(
    save_path,
    stable_diffusion_path,
    
    sheet_type,
    writers,
    
    device,
):
    char_stylist = CharStylist.load(
        save_path,
        stable_diffusion_path,
        
        device,
    )
    
    if sheet_type == "hiragana":
        nrow = 11
        chars = [
            "あ", "い", "う", "え", "お",
            "か", "き", "く", "け", "こ",
            "さ", "し", "す", "せ", "そ",
            "た", "ち", "つ", "て", "と",
            "な", "に", "ぬ", "ね", "の",
            "は", "ひ", "ふ", "へ", "ほ",
            "ま", "み", "む", "め", "も",
            "や", None, "ゆ", None, "よ",
            "ら", "り", "る", "れ", "ろ",
            "わ", "ゐ", None, "ゑ", "を",
            "ん", None, None, None, None,
        ]
        chars = [chars[(11 - (i % 11 + 1)) * 5 + (i // 11)] for i in range(len(chars))]
        
    elif sheet_type == "katakana":
        nrow = 11
        chars = [
            "ア", "イ", "ウ", "エ", "オ",
            "カ", "キ", "ク", "ケ", "コ",
            "サ", "シ", "ス", "セ", "ソ",
            "タ", "チ", "ツ", "テ", "ト",
            "ナ", "ニ", "ヌ", "ネ", "ノ",
            "ハ", "ヒ", "フ", "ヘ", "ホ",
            "マ", "ミ", "ム", "メ", "モ",
            "ヤ", None, "ユ", None, "ヨ",
            "ラ", "リ", "ル", "レ", "ロ",
            "ワ", "ヰ", None, "ヱ", "ヲ",
            "ン", None, None, None, None,
        ]
        chars = [chars[(11 - (i % 11 + 1)) * 5 + (i // 11)] for i in range(len(chars))]
    
    else:
        raise Exception(f"unknown sheet_type: {sheet_type}")
    
    for char in chars:
        if char is None:
            continue
        if char not in char_stylist.char2idx:
            raise Exception(f"unknown character: {char}")
    
    for writer in writers:
        if writer not in char_stylist.writer2idx:
            raise Exception(f"unknown writer: {writer}")
    
    # sampling
    images_list = []
    for char in tqdm(chars):
        if char is None:
            images_list.append(torch.zeros((len(writers), 3, 64, 64)))
        else:
            images_list.append(char_stylist.sampling(char, writers))
    
    for i, writer in enumerate(writers):
        image_path = os.path.join(save_path, "generated", f"sheet_{sheet_type} writer={writer}.jpg")
        save_images([images[i] for images in images_list], image_path, nrow=nrow)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_path", type=pathstr, default=pathstr("./datadisk/save_path/test ETL4,ETL5"))
    parser.add_argument("--stable_diffusion_path", type=pathstr, default=pathstr("~/datadisk/stable-diffusion-v1-5"))
    
    parser.add_argument("--sheet_type", type=str, default="hiragana")
    parser.add_argument("--writers", type=str, nargs="*",
                        default=([f"ETL4_{i}" for i in range(5001, 5016 + 1)] + [f"ETL5_{i}" for i in range(6001, 6016 + 1)]))
    
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    # todo: validation
    
    main(
        args.save_path,
        args.stable_diffusion_path,
        
        args.sheet_type,
        args.writers,
        
        args.device,
    )
