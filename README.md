<h1 align="center">⚡️- Image<br><sub><sup>An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer</sup></sub></h1>

<div align="center">

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://tongyi-mai.github.io/Z-Image-blog/)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-Z--Image--Turbo-yellow)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Online_Demo-Z--Image--Turbo-blue)](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo)&#160;
[![ModelScope Model](https://img.shields.io/badge/🤖%20Checkpoint-Z--Image--Turbo-624aff)](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)&#160;
[![ModelScope Space](https://img.shields.io/badge/🤖%20Online_Demo-Z--Image--Turbo-17c7a7)](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=469191&modelType=Checkpoint&sdVersion=Z_IMAGE_TURBO&modelUrl=modelscope%253A%252F%252FTongyi-MAI%252FZ-Image-Turbo%253Frevision%253Dmaster%7D%7BOnline)&#160;
[![Art Gallery PDF](https://img.shields.io/badge/%F0%9F%96%BC%20Art_Gallery-PDF-ff69b4)](assets/Z-Image-Gallery.pdf)&#160;
[![Web Art Gallery](https://img.shields.io/badge/%F0%9F%8C%90%20Web_Art_Gallery-online-00bfff)](https://modelscope.cn/studios/Tongyi-MAI/Z-Image-Gallery/summary)&#160;
<a href="https://arxiv.org/abs/2511.22699" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv" height="21px"></a>


Welcome to the official repository for the Z-Image（造相）project!

</div>



## ✨ Z-Image

Z-Image is a powerful and highly efficient image generation model with **6B** parameters. Currently there are three variants:

- 🚀 **Z-Image-Turbo** – A distilled version of Z-Image that matches or exceeds leading competitors with only **8 NFEs** (Number of Function Evaluations). It offers **⚡️sub-second inference latency⚡️** on enterprise-grade H800 GPUs and fits comfortably within **16G VRAM consumer devices**. It excels in photorealistic image generation, bilingual text rendering (English & Chinese), and robust instruction adherence.

- 🧱 **Z-Image-Base** – The non-distilled foundation model. By releasing this checkpoint, we aim to unlock the full potential for community-driven fine-tuning and custom development.

- ✍️ **Z-Image-Edit** – A variant fine-tuned on Z-Image specifically for image editing tasks. It supports creative image-to-image generation with impressive instruction-following capabilities, allowing for precise edits based on natural language prompts.

### 📣 News

*   **[2025-12-08]** 🏆 Z-Image-Turbo ranked 8th overall on the **Artificial Analysis Text-to-Image Leaderboard**, making it the 🥇 <strong style="color: #FFC300;">#1 open-source model</strong>! [Check out the full leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image).
*   **[2025-12-01]** 🎉 Our technical report for Z-Image is now available on [arXiv](https://arxiv.org/abs/2511.22699).
*   **[2025-11-26]** 🔥 **Z-Image-Turbo is released!** We have released the model checkpoint on [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and [ModelScope](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo). Try our [online demo](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo)!

### 📥 Model Zoo

| Model | Pre-Training | SFT | RL | Step | CFG | Task | Visual Quality | Diversity | Fine-Tunability | Hugging Face | ModelScope |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Z-Image-Omni-Base** | ✅ | ❌ | ❌ | 50 | ✅ | Gen. / Editing | Medium | High | Easy | *To be released* | *To be released* |
| **Z-Image** | ✅ | ✅ | ❌ | 50 | ✅ | Gen. | High | Medium | Easy | *To be released* | *To be released* |
| **Z-Image-Turbo** | ✅ | ✅ | ✅ | 8 | ❌ | Gen. | Very High | Low | Difficult | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint%20-Z--Image--Turbo-yellow)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) <br> [![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Online%20Demo-Z--Image--Turbo-blue)](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo) | [![ModelScope Model](https://img.shields.io/badge/🤖%20%20Checkpoint-Z--Image--Turbo-624aff)](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) <br> [![ModelScope Space](https://img.shields.io/badge/%F0%9F%A4%96%20Online%20Demo-Z--Image--Turbo-17c7a7)](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=469191&modelType=Checkpoint&sdVersion=Z_IMAGE_TURBO&modelUrl=modelscope%3A%2F%2FTongyi-MAI%2FZ-Image-Turbo%3Frevision%3Dmaster) |
| **Z-Image-Edit** | ✅ | ✅ | ❌ | 50 | ✅ | Editing | High | Medium | Easy | *To be released* | *To be released* |

The figure below illustrates at which training stage each model is produced.

![Training Pipeline of Z-Image](assets/training_pipeline.jpg)

### 🖼️ Showcase

📸 **Photorealistic Quality**: **Z-Image-Turbo** delivers strong photorealistic image generation while maintaining excellent aesthetic quality.

![Showcase of Z-Image on Photo-realistic image Generation](assets/showcase_realistic.png)

📖 **Accurate Bilingual Text Rendering**: **Z-Image-Turbo** excels at accurately rendering complex Chinese and English text.

![Showcase of Z-Image on Bilingual Text Rendering](assets/showcase_rendering.png)

💡  **Prompt Enhancing & Reasoning**: Prompt Enhancer empowers the model with reasoning capabilities, enabling it to transcend surface-level descriptions and tap into underlying world knowledge.

![reasoning.jpg](assets/reasoning.png)

🧠 **Creative Image Editing**: **Z-Image-Edit** shows a strong understanding of bilingual editing instructions, enabling imaginative and flexible image transformations.

![Showcase of Z-Image-Edit on Image Editing](assets/showcase_editing.png)

### 🏗️ Model Architecture
We adopt a **Scalable Single-Stream DiT** (S3-DiT) architecture. In this setup, text, visual semantic tokens, and image VAE tokens are concatenated at the sequence level to serve as a unified input stream, maximizing parameter efficiency compared to dual-stream approaches.

![Architecture of Z-Image and Z-Image-Edit](assets/architecture.webp)

### 📈 Performance

Z-Image-Turbo's performance has been validated on multiple independent benchmarks, where it consistently demonstrates state-of-the-art results, especially as the leading open-source model.

#### Artificial Analysis Text-to-Image Leaderboard
On the highly competitive [Artificial Analysis Leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image), Z-Image-Turbo ranked **8th overall** and secured the top position as the 🥇 <strong style="color: gold;">#1 Open-Source Model</strong>, outperforming all other open-source alternatives.


<p align="center">
  <a href="https://artificialanalysis.ai/image/leaderboard/text-to-image">
    <img src="assets/image_arena_all.jpg" alt="Z-Image Rank on Artificial Analysis Leaderboard"/><br />
    <span style="font-size:1.05em; cursor:pointer; text-decoration:underline;"> Artificial Analysis Leaderboard</span>
  </a>
</p>

<p align="center">
  <a href="https://artificialanalysis.ai/image/leaderboard/text-to-image">
    <img src="assets/image_arena_os.jpg" alt="Z-Image Rank on Artificial Analysis Leaderboard (Open-Source Model Only)"/><br />
    <span style="font-size:1.05em; cursor:pointer; text-decoration:underline;"> Artificial Analysis Leaderboard (Open-Source Model Only)</span>
  </a>
</p>

#### Alibaba AI Arena Text-to-Image Leaderboard
According to the Elo-based Human Preference Evaluation on [*Alibaba AI Arena*](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=T2I), Z-Image-Turbo also achieves state-of-the-art results among open-source models and shows highly competitive performance against leading proprietary models.

<p align="center">
  <a href="https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=T2I">
    <img src="assets/leaderboard.png" alt="Z-Image Elo Rating on AI Arena"/><br />
    <span style="font-size:1.05em; cursor:pointer; text-decoration:underline;"> Alibaba AI Arena Text-to-Image Leaderboard</span>
  </a>
</p>


### 🚀 Quick Start
#### (1) PyTorch Native Inference
Build a virtual environment you like and then install the dependencies:
```bash
pip install -e .
```
Then run the following code to generate an image:
```bash
python inference.py
```

#### (2) Diffusers Inference
Install the latest version of diffusers, use the following command:
<details>
  <summary>Click here for details for why you need to install diffusers from source</summary>

  We have submitted two pull requests ([#12703](https://github.com/huggingface/diffusers/pull/12703) and [#12715](https://github.com/huggingface/diffusers/pull/12715)) to the 🤗 diffusers repository to add support for Z-Image. Both PRs have been merged into the latest official diffusers release.
  Therefore, you need to install diffusers from source for the latest features and Z-Image support.

</details>

```bash
pip install git+https://github.com/huggingface/diffusers
```

Then, try the following code to generate an image:
```python
import torch
from diffusers import ZImagePipeline

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

# 2. Generate Image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("example.png")
```

## 🔬 Decoupled-DMD: The Acceleration Magic Behind Z-Image

[![arXiv](https://img.shields.io/badge/arXiv-2511.22677-b31b1b.svg)](https://arxiv.org/abs/2511.22677)

Decoupled-DMD is the core few-step distillation algorithm that empowers the 8-step Z-Image model.

Our core insight in Decoupled-DMD  is that the success of existing DMD (Distributaion Matching Distillation) methods is the result of two independent, collaborating mechanisms:

-   **CFG Augmentation (CA)**: The primary **engine** 🚀 driving the distillation process, a factor largely overlooked in previous work.
-   **Distribution Matching (DM)**: Acts more as a **regularizer** ⚖️, ensuring the stability and quality of the generated output.

By recognizing and decoupling these two mechanisms, we were able to study and optimize them in isolation. This ultimately motivated us to develop an improved distillation process that significantly enhances the performance of few-step generation.

![Diagram of Decoupled-DMD](assets/decoupled-dmd.webp)

## 🤖 DMDR: Fusing DMD with Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2511.13649-b31b1b.svg)](https://arxiv.org/abs/2511.13649)

Building upon the strong foundation of Decoupled-DMD, our 8-step Z-Image model has already demonstrated exceptional capabilities. To achieve further improvements in terms of semantic alignment, aesthetic quality, and structural coherence—while producing images with richer high-frequency details—we present **DMDR**.

Our core insight behind DMDR is that Reinforcement Learning (RL) and Distribution Matching Distillation (DMD) can be synergistically integrated during the post-training of few-step models. We demonstrate that:

-   **RL Unlocks the Performance of DMD** 🚀
-   **DMD Effectively Regularizes RL** ⚖️

![Diagram of DMDR](assets/DMDR.webp)

## 🎉 Community Works

- [Cache-DiT](https://github.com/vipshop/cache-dit) provides inference acceleration for **Z-Image** and **Z-Image-ControlNet** via DBCache, Context Parallelism and Tensor Parallelism. It achieves nearly **4x** speedup on 4 GPUs with negligible precision loss. Please visit their [example](https://github.com/vipshop/cache-dit/blob/main/examples) for more details.
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) is a pure C++ diffusion model inference engine that supports fast and memory-efficient Z-Image inference across multiple platforms (CUDA, Vulkan, etc.). You can use stable-diffusion.cpp to generate images with Z-Image on machines with as little as 4GB of VRAM. For more information, please refer to [How to Use Z‐Image on a GPU with Only 4GB VRAM](https://github.com/leejet/stable-diffusion.cpp/wiki/How-to-Use-Z%E2%80%90Image-on-a-GPU-with-Only-4GB-VRAM).
- [LeMiCa](https://github.com/UnicomAI/LeMiCa) provides a training-free, timestep-level acceleration method that conveniently speeds up Z-Image inference. For more details, see [LeMiCa4Z-Image](https://github.com/UnicomAI/LeMiCa/tree/main/LeMiCa4Z-Image).
- [ComfyUI ZImageLatent](https://github.com/HellerCommaA/ComfyUI-ZImageLatent) provdes an easy to use latent of the official Z-Image resolutions.
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) has provided more support for Z-Image, including LoRA training, full training, distillation training, and low-VRAM inference. Please refer to the [document](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md) of DiffSynth-Studio.
- [vllm-omni](https://github.com/vllm-project/vllm-omni), a framework that extends its support for omni-modality model fast inference and serving, now [supports](https://github.com/vllm-project/vllm-omni/blob/main/docs/models/supported_models.md) Z-Image.
- [SGLang-Diffusion](https://lmsys.org/blog/2025-11-07-sglang-diffusion/) brings SGLang's state-of-the-art performance to accelerate image and video generation for diffusion models, now [supporting](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/zimage_pipeline.py) Z-Image.


## 🚀 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tongyi-MAI/Z-Image&type=date&legend=top-left)](https://www.star-history.com/#Tongyi-MAI/Z-Image&type=date&legend=top-left)


## 📜 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{team2025zimage,
  title={Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer},
  author={Z-Image Team},
  journal={arXiv preprint arXiv:2511.22699},
  year={2025}
}

@article{liu2025decoupled,
  title={Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield},
  author={Dongyang Liu and Peng Gao and David Liu and Ruoyi Du and Zhen Li and Qilong Wu and Xin Jin and Sihan Cao and Shifeng Zhang and Hongsheng Li and Steven Hoi},
  journal={arXiv preprint arXiv:2511.22677},
  year={2025}
}

@article{jiang2025distribution,
  title={Distribution Matching Distillation Meets Reinforcement Learning},
  author={Jiang, Dengyang and Liu, Dongyang and Wang, Zanyi and Wu, Qilong and Jin, Xin and Liu, David and Li, Zhen and Wang, Mengmeng and Gao, Peng and Yang, Harry},
  journal={arXiv preprint arXiv:2511.13649},
  year={2025}
}

```

## 🤝 We're Hiring!

We're actively looking for **Research Scientists**, **Engineers**, and **Interns** to work on foundational generative models and their applications. Interested candidates please send your resume to: **jingpeng.gp@alibaba-inc.com**
