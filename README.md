# CharStylist

submodule として stable-diffusion を利用  
`-` を含むディレクトリは `import` できない（Syntax Error になる）ので `_` に置き換えている

## 学習

```
python train.py
```

## サンプリング

```
python sampling
```

```
python sheet_sampling
```

## WordStylist からの変更点

- 位置埋め込みを削除
- 損失換数に $`L_div`$ のようなものを追加（最大化する）
  
  Zeng+, [Diversity Regularized StarGAN for Multi-style Fonts Generation of Chinese Characters]((https://iopscience.iop.org/article/10.1088/1742-6596/1880/1/012017/pdf)), J. Phys.: Conf. Ser. 1880 012017, 2021.  
  $$\mathcal{L}_{div}(G) = \mathbb{E}_{x, x', c} \left[ \| G(x, c) - G(x', c) \|_1 / \| x - x' \|_1 \right]$$

## WordStylist のバグ？

- Downsample で呼び出す AvgPool2d の引数が変
  https://github.com/koninik/WordStylist/blob/f18522306e533a01eb823dc4369a4bcb7ea67bcc/unet.py#L420
  `torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)` の `kernel_size` を 2 回指定することになる
  
  stable-diffusion から移植するときに誤ったと思われる
  https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/openaimodel.py#L134
  
  CharStylist では stable-diffusion の方を呼び出すようにした
  