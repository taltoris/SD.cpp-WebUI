# SD.cpp-WebUI

A lightweight, browser-based web interface for **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** â€” powered by Flask and designed for **fast, local, low-resource image generation**.

![SD.cpp-WebUI Screenshot](https://via.placeholder.com/800x500?text=SD.cpp-WebUI+Interface)  
*(Screenshot placeholder â€” replace with actual image)*

---

## í ½íº€ Features

**âœ… Tested & Working**
- Text-to-Image (Txt2Img) generation  
- Built-in gallery for generated outputs  
- Full support for **Z-Image**, **SD3.5**, and **Flux** models  
- Executes inference through `sd-cli` from **stable-diffusion.cpp**

**âš ï¸ Experimental / Untested**
- Image-to-Image (Img2Img) pipeline  
- Video generation (WAN models) â€” may require additional dependencies  

**í ½í´§ Advanced Options**
- VAE tiling, Flash Attention, CPU offload  
- LoRA and embeddings support (when supported by model backend)

---

## í ½í» ï¸ Requirements

- Docker + Docker Compose *(recommended environment)*
- NVIDIA GPU with CUDA support *(enabled by default in Dockerfile)*
- `stable-diffusion.cpp` built and accessible in the same parent directory

---

## í ½í³¦ Installation

### 1. Clone both repositories
```
git clone https://github.com/taltoris/SD.cpp-WebUI
git clone https://github.com/leejet/stable-diffusion.cpp
```

### 2. Build `stable-diffusion.cpp`
Follow the official guide:  
í ½í±‰ [Build Instructions](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md)

Then prepare model directories:
```
cd stable-diffusion.cpp
mkdir -p models/{clip,diffusion,llm,t5,text_encoders,vae}
```

Place your models (e.g. `.gguf`, `.safetensors`) in the appropriate subfolders, such as `models/diffusion/`.

### 3. Run the WebUI
```
cd ../SD.cpp-WebUI
docker compose up
```

Then open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## í ½í³ Directory Structure

```
SD.cpp-WebUI/
â”œâ”€â”€ app.py              # Flask backend
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ docker-compose.yml
â”œâ”€â”€ config.json         # Preset model configurations
â”œâ”€â”€ models/             # Drop model files here if not using sibling repo
â”œâ”€â”€ templates/          # HTML templates
â”œâ”€â”€ static/             # CSS, JS, and frontend assets
â””â”€â”€ output/             # Generated image output

stable-diffusion.cpp/
â””â”€â”€ models/
    â”œâ”€â”€ clip/
    â”œâ”€â”€ diffusion/
    â”œâ”€â”€ llm/
    â”œâ”€â”€ t5/
    â”œâ”€â”€ text_encoders/
    â””â”€â”€ vae/
```

---

## í ½í³š Models

Download compatible models from **[Hugging Face](https://huggingface.co)** or **[CivitAI](https://civitai.com)** in **GGUF** or **Safetensors** format.

**Tested models:**
| Model | File Example | Notes |
|-------|---------------|-------|
| Z-Image | `z_image_turbo-Q8_0.gguf` | Very fast generation |
| SD3.5 | `stable-diffusion-v3-5-medium-pure-Q4_0.gguf` | Balanced quality-speed |
| Flux | `flux1-dev-q4_0.gguf` | Excellent aesthetics |

Place models in the corresponding folder under `stable-diffusion.cpp/models/`.

---

## í ½í²¡ Built With

- í ¾í·  [**stable-diffusion.cpp**](https://github.com/leejet/stable-diffusion.cpp) â€” C++ inference engine  
- í ¾í·© Flask â€” Lightweight Python web server  
- í ½í²» JavaScript + CSS â€” Responsive and minimal UI  
- í ½í°‹ Docker â€” Easy, isolated deployment  

---

## í ¾í´ Contributing

Pull requests are welcome!  
Open an issue for bug reports, feature ideas, or general discussion.

---

## í ½í³œ License

**MIT License** â€” feel free to fork, modify, and build upon.

