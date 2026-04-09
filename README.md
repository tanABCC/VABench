## VABench: A Comprehensive Benchmark for Audio-Video Generation
<!-- [![License](https://img.shields.io/github/license/yourname/AAA)](LICENSE) -->


### News
* **`Dec. 9th, 2025`:** We’ve released our paper on arXiv, along with our test data!
* **`Apr. 9th, 2026`:** We release our dataset and code.
  

### 📄 Paper has been released on arXiv
https://arxiv.org/abs/2512.09299

### Dataset
test images:https://huggingface.co/datasets/tanABBCC/VABench_I2AV/tree/main

model cache:https://huggingface.co/datasets/tanABBCC/VABENCH_CACHE_DIR/tree/main

### Abstract
Recent advances in video generation have been remarkable, enabling models to produce visually compelling videos with synchronized audio. While existing video generation benchmarks provide comprehensive metrics for visual quality, they lack convincing evaluations for audio-video generation, especially for models aiming to generate synchronized audio-video outputs. To address this gap, we introduce VABench, a comprehensive and multi-dimensional benchmark framework designed to systematically evaluate the capabilities of synchronous audio-video generation. VABench encompasses three primary task types: text-to-audio-video (T2AV), image-to-audio-video (I2AV), and stereo audio-video generation. It further establishes two major evaluation modules covering 15 dimensions. These dimensions specifically assess pairwise similarities (text-video, text-audio, video-audio), audio-video synchronization, lip-speech consistency, and carefully curated audio and video question-answering (QA) pairs, among others. Furthermore, VABench covers seven major content categories: animals, human sounds, music, environmental sounds, synchronous physical sounds, complex scenes, and virtual worlds. We provide a systematic analysis and visualization of the evaluation results, aiming to establish a new standard for assessing video generation models with synchronous audio capabilities and to promote the comprehensive advancement of the field.

### Data Preparation
Please note that the videos generated need to be named the same as in the first column of the mapping file.
### How to use:
```
conda create -n vabench python=3.11
conda activate vabench
pip install nisqa==2.0.post2
pip install -r requirement.txt
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/facebookresearch/ImageBind.git
```
To run successfully, at least two GPUs with 80GB of VRAM are recommended.

After the date being prepared:
```
export VABENCH_CACHE_DIR = /path/to/the/cache/dir
python evaluate.py --data_dir /path/to/the/test --output_dir /path/to/the/result/dir
```

###  Citation

Please cite our paper if you find our work helpful.

```
@article{hua2025vabench,
  title={VABench: A Comprehensive Benchmark for Audio-Video Generation},
  author={Hua, Daili and Wang, Xizhi and Zeng, Bohan and Huang, Xinyi and Liang, Hao and Niu, Junbo and Chen, Xinlong and Xu, Quanqing and Zhang, Wentao},
  journal={arXiv preprint arXiv:2512.09299},
  year={2025}
}
```