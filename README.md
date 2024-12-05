# HunyuanVideo-xDiT: Parallel Inference for Hunyuan Video Generation Model

This repository provides an parallel approach to delpoy the Video Generation Model [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) using Unified Sequence Parallelism (USP) by [xDiT](https://github.com/xdit-project/xDiT).
xDiT is a Scalable Inference Engine for Diffusion Transformers (DiTs) on multi-GPU Clusters.
It has successfully provide low latency infernece solution for a varitey of DiTs models, including mochi-1, CogVideoX, Flux.1, SD3, etc.

## Download Checkpoint

Follow the [instruction](https://github.com/Tencent/HunyuanVideo?tab=readme-ov-file#-download-pretrained-models) to download the checkpoint.

## Install

```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Usage

```bash
bash run.sh
```

## BibTeX
If you find [HunyuanVideo](https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{kong2024hunyuanvideo,
      title={HunyuanVideo: A Systematic Framework For Large Video Generative Models}, 
      author={Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Junkun Yuan, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yanxin Long, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, and Jie Jiang, along with Caesar Zhong},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

If you find [xDiT](https://arxiv.org/abs/2411.01738) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{kong2024hunyuanvideo,
@article{fang2024xdit,
  title={xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism},
  author={Fang, Jiarui and Pan, Jinzhe and Sun, Xibo and Li, Aoyu and Wang, Jiannan},
  journal={arXiv preprint arXiv:2411.01738},
  year={2024}
}
```