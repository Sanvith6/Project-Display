Real-Time Cricket Event Interpretation and Commentary Synthesis
An advanced AI-powered system that generates high-energy, professional-style cricket commentary in real-time. By combining state-of-the-art computer vision models, automated scorecard OCR, and LLM-driven natural language generation, this project transforms raw cricket video into a broadcast-ready experience with synthesized voice commentary.

Cricket AI Banner

🌟 Key Features
Multi-Modal AI Pipeline: Integrates YOLO object detection, CNN-based classifiers, and R(2+1)D spatio-temporal video models.
YOLO-First Optimization: Triggers heavy classification models (Shot, Umpire, Runout) only when relevant objects are detected, saving 40%+ computational load.
Smart Scorecard OCR: Automated extraction of runs, wickets, and overs using adaptive image preprocessing and OCR.Space integration.
Dynamic Commentary Generation: Context-aware narratives powered by LLMs (Jina DeepSearch) that strictly follow match progression.
Pro TTS Synthesis: High-quality audio generation using ElevenLabs with automatic fallback to Microsoft Edge TTS.
Interactive Match Analysis: REST API for real-time video upload, status tracking, and timestamp-based Q&A with an AI analyst.
🏗 System Architecture
The system follows a 7-stage modular pipeline designed for scalability and precision:


🛠 Tech Stack
Category	Technologies
Backend	Python, FastAPI, Uvicorn, Asyncio
Deep Learning	PyTorch, Torchvision, Ultralytics (YOLOv8), Timm
Computer Vision	OpenCV, PIL (Pillow)
NLP & LLM	OpenAI API (Jina DeepSearch compatible), Prompt Engineering
Audio/TTS	ElevenLabs API, Edge-TTS, FFmpeg
Frontend	HTML5, CSS3 (Glassmorphism), JavaScript (Vanilla)
🚀 Getting Started
Prerequisites
Python 3.10+
FFmpeg: Required for video/audio processing.
CUDA-Capable GPU (Optional): Recommended for faster inference.
Installation
Clone the repository:

git clone https://github.com/Sanvith6/Project-Display.git
cd Project-Display
Create a virtual environment:

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
Configuration
Create a .env file in the root directory and add your API keys:

JINA_API_KEY=your_jina_key
OCRSPACE_API_KEY=your_ocr_space_key
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id_optional
Running the Application
Start the FastAPI server:

python -m uvicorn app:app --reload
Access the Web Interface: Open http://localhost:8000 in your browser.

📁 Project Structure
├── app.py              # FastAPI Web Server & REST Endpoints
├── commentator.py      # Core Pipeline Orchestrator
├── inference.py        # Optimized Visual AI Pipeline (YOLO-First)
├── ocr.py              # Scorecard Processing & Text Parsing
├── llm.py              # Commentary Generation Logic
├── tts.py              # Voice Synthesis (ElevenLabs & Edge TTS)
├── models.py           # PyTorch Model Loaders & Architectures
├── config.py           # Global Configuration & Paths
├── static/             # Responsive Web Frontend
├── models/             # Pretrained Model Weights (.pt, .pth)
└── requirements.txt    # Project Dependencies
🎯 Future Roadmap
 Real-time livestream processing support.
 Support for multiple commentary languages.
 Integration with more sports (Football, Tennis, etc.).
 Mobile app for real-time match notifications.
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

Developed with ❤️ for the Cricket Community.
