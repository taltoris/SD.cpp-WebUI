Here's your content beautifully formatted as a clean, polished Markdown document — with improved structure, consistent emoji usage, and visual hierarchy for readability:

---

# SD.cpp-WebUI

A lightweight, browser-based web interface for **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** — powered by Flask and designed for **fast, local, low-resource image generation**.

---

## ✅ Features

### **Tested & Working**
- Text-to-Image (Txt2Img) generation  
- Built-in gallery for generated outputs  
- Full support for **Z-Image**, **SD3.5**, and **Flux** models  
- Executes inference through `sd-cli` from **stable-diffusion.cpp**

### **⚠️ Experimental / Untested**
- Image-to-Image (Img2Img) pipeline  
- Video generation (WAN models) — may require additional dependencies  

### **⚙️ Advanced Options**
- VAE tiling, Flash Attention, CPU offload  
- LoRA and embeddings support (when supported by model backend)

---

## 📦 Requirements

- **Docker + Docker Compose** *(recommended environment)*  
- **NVIDIA GPU with CUDA support** *(enabled by default in Dockerfile)*  
- `stable-diffusion.cpp` built and accessible in the same parent directory

---

## 🚀 Installation

### 1. Clone both repositories
```bash
git clone https://github.com/taltoris/SD.cpp-WebUI
git clone https://github.com/leejet/stable-diffusion.cpp
```

### 2. Build `stable-diffusion.cpp`
Follow the official guide:  
🔗 [Build Instructions](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md)

Then prepare model directories:
```bash
cd stable-diffusion.cpp
mkdir -p models/{clip,diffusion,llm,t5,text_encoders,vae}
```

Place your models (e.g., `.gguf`, `.safetensors`) in the appropriate subfolders — e.g., `models/diffusion/`.

### 3. Run the WebUI
```bash
cd ../SD.cpp-WebUI
docker compose up
```

Then open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 📁 Directory Structure

```
SD.cpp-WebUI/
├── app.py              # Flask backend
├── Dockerfile
├── docker-compose.yml
├── config.json         # Preset model configurations
├── models/             # Drop model files here if not using sibling repo
├── templates/          # HTML templates
├── static/             # CSS, JS, and frontend assets
└── output/             # Generated image output

stable-diffusion.cpp/
└── models/
    ├── clip/
    ├── diffusion/
    ├── llm/
    ├── t5/
    ├── text_encoders/
    └── vae/
```

---

## 🖼️ Models

Download compatible models from **[Hugging Face](https://huggingface.co)** or **[CivitAI](https://civitai.com)** in **GGUF** or **Safetensors** format.

### ✅ Tested Models

| Model       | File Example                          | Notes                     |
|-------------|----------------------------------------|---------------------------|
| Z-Image     | `z_image_turbo-Q8_0.gguf`              | Very fast generation      |
| SD3.5       | `stable-diffusion-v3-5-medium-pure-Q4_0.gguf` | Balanced quality-speed |
| Flux        | `flux1-dev-q4_0.gguf`                  | Excellent aesthetics      |

> Place models in the corresponding folder under `stable-diffusion.cpp/models/`.

---

## 💡 Built With

- 🧠 [**stable-diffusion.cpp**](https://github.com/leejet/stable-diffusion.cpp) — C++ inference engine  
- 🐍 **Flask** — Lightweight Python web server  
- 🎨 **JavaScript + CSS** — Responsive and minimal UI  
- 🐳 **Docker** — Easy, isolated deployment  

---

## 🤝 Contributing

Pull requests are welcome!  
Open an issue for bug reports, feature ideas, or general discussion.

---

## 📜 License

**MIT License** — feel free to fork, modify, and build upon.
