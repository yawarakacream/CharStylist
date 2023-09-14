import os
import argparse

from torch.utils.data import ConcatDataset, DataLoader

import torchvision

from char_stylist import CharStylist
from dataset import EtlcdbDataset
from utility import pathstr


def main(
    save_path,
    stable_diffusion_path,
    
    image_size,
    dim_char_embedding,
    num_res_blocks,
    num_heads,
    
    learning_rate,
    ema_beta,
    diffusion_noise_steps,
    diffusion_beta_start,
    diffusion_beta_end,
    
    batch_size,
    epochs,
    num_workers,
    
    corpuses,
    etlcdb_path,
    
    test_chars,
    test_writers,
    
    device,
):
    # 謎
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # 各データセットに渡して更新させる
    char2idx = {}
    writer2idx = {}
    
    datasets = []
    
    for corpus in corpuses:
        corpus_type = corpus[0]
        
        if corpus_type == "etlcdb":
            _, etlcdb_process_type, etlcdb_names = corpus
            datasets.append(EtlcdbDataset(char2idx, writer2idx, etlcdb_path, etlcdb_process_type, etlcdb_names, transforms))
            
        else:
            raise Exception(f"unknown corpus type: {corpus_type}")
            
    data_loader = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    print("initializing CharStylist")
    char_stylist = CharStylist(
        save_path,
        stable_diffusion_path,
        
        char2idx,
        writer2idx,
        
        image_size,
        dim_char_embedding,
        num_res_blocks,
        num_heads,
        
        learning_rate,
        ema_beta,
        diffusion_noise_steps,
        diffusion_beta_start,
        diffusion_beta_end,
        
        device,
    )
    print("initialized")
    
    print("training started")
    char_stylist.train(data_loader, epochs, test_chars, test_writers)
    print("training finished")


def corpus(data: str):
    data = data.split("/")
    
    corpus_type = data[0]
    
    if corpus_type == "etlcdb":
        if len(data) != 3:
            raise Exception(f"illegal etlcdb: {data}")
        
        etlcdb_process_type = data[1]
        etlcdb_names = data[2].split(",")
        return corpus_type, etlcdb_process_type, etlcdb_names
    
    else:
        raise Exception(f"unknown corpus type: {corpus_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_path", type=pathstr, default=pathstr("./datadisk/save_path/test"))
    parser.add_argument("--stable_diffusion_path", type=pathstr, default=pathstr("~/datadisk/stable-diffusion-v1-5"))
    
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--dim_char_embedding", type=int, default=320)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--ema_beta", type=float, default=0.995)
    parser.add_argument("--diffusion_noise_steps", type=int, default=1000)
    parser.add_argument("--diffusion_beta_start", type=float, default=0.0001)
    parser.add_argument("--diffusion_beta_end", type=float, default=0.02)
    
    parser.add_argument("--batch_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    
    parser.add_argument("--corpuses", type=corpus, nargs="*", default=[corpus("etlcdb/no_background inversed 64x64/ETL4,ETL5")])
    parser.add_argument("--etlcdb_path", type=pathstr, default=pathstr("~/datadisk/etlcdb_processed"))
    
    parser.add_argument("--test_chars", type=str, nargs="*", default=["あ", "く", "さ", "ほ", "ア", "コ", "テ", "ホ"])
    parser.add_argument("--test_writers", type=str, nargs="*",
                        default=([f"ETL4_500{i}" for i in range(1, 8 + 1)] + [f"ETL5_600{i}" for i in range(1, 8 + 1)]))
    
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    # todo: validation
    
    main(
        args.save_path,
        args.stable_diffusion_path,
        
        args.image_size,
        args.dim_char_embedding,
        args.num_res_blocks,
        args.num_heads,
        
        args.learning_rate,
        args.ema_beta,
        args.diffusion_noise_steps,
        args.diffusion_beta_start,
        args.diffusion_beta_end,
        
        args.batch_size,
        args.epochs,
        args.num_workers,
        
        args.corpuses,
        args.etlcdb_path,
        
        args.test_chars,
        args.test_writers,
        
        args.device,
    )
