# 🌌 Chandra Prakash Bathula

🎓 **Adjunct Faculty & Software Developer @ Saint Louis University**  
💻 **Frontend Engineer | AI Researcher | ML Developer | Educator**

Welcome to my GitHub! I'm a passionate **AI Developer** and **Frontend Engineer** with a **M.S. in Information Systems (GPA: 3.93/4.0)** from Saint Louis University. I craft intelligent, scalable web systems blending **Machine Learning** and **modern frontend development**. I've mentored **300+ students**, deployed **40+ full-stack projects**, and built solutions with **React.js**, **Python**, **AWS**, and **ML models**.

*“Technology is a language that connects intelligence, creativity, and purpose.”* 🌠

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?logo=javascript&logoColor=black)
![AWS](https://img.shields.io/badge/-AWS-232F3E?logo=amazonaws&logoColor=white)
![Hugging Face](https://img.shields.io/badge/-Hugging%20Face-F9AB00?logo=huggingface&logoColor=white)
![GitHub followers](https://img.shields.io/github/followers/ChandraPrakash-Bathula?style=social)

---

## 🧠 About Me

- 🎓 **M.S. in Information Systems**, Saint Louis University (GPA: 3.93/4.0)
- 🧑‍🏫 Mentored 300+ students in **Software Development**, **Advanced Software Development**, and **Applied Analytics**
- 🚀 Built internal SLU web apps and automated iPaaS workflows, cutting manual workload by **40%** and boosting performance by **30%**
- 🧠 Researched **ML/NLP** with **Word2Vec**, **XGBoost**, **CNNs**, and **PyTorch** for scalable, interpretable systems
- ⚡ **Fun Fact**: I love turning complex data into 3D visualizations that feel out-of-this-world!

---

## 🚀 What I Do

### 🧩 Software Development & Teaching
- Guided 20+ capstone projects using **React.js**, **Node.js**, **Python**, **AWS**, and **BigQuery**
- Deployed production-grade apps via **Vercel**, **Firebase**, and **AWS**
- Taught scalable software design and ML integration to aspiring developers

### 💻 Engineering & Research
- Developed web apps with **React.js**, **Vue.js**, **Python (Flask/Django)**, and **AWS (Lambda, S3, RDS, EC2)**
- Built analytics dashboards for real-time insights
- Advanced **ML/NLP** projects with **Word2Vec**, **TF-IDF**, and **CNNs**

---

## 🧩 Featured Projects

| Project | Tech Stack | Highlights |
|:--------|:-----------|:-----------|
| [**CIFAR-10 CNN Model**](https://huggingface.co/chandu1617/CIFAR10-CNN_Model) | PyTorch, CNN, Hugging Face | Achieved **92.59% test accuracy** on CIFAR-10 with 9 convolutional layers. [Demo](https://huggingface.co/spaces/chandu1617/cifar10-cnn-demo) |
| [**MoodFlix / VizFlix**](https://viz-flix-gpt.vercel.app/) | React.js, Redux, OpenAI API, TailwindCSS | GPT-powered movie recommendations, boosting engagement by 30% |
| [**TubeFlix**](https://utubeflix-79845.web.app/) | React.js, TailwindCSS, Firebase | Adaptive video streaming for 1K+ videos, 30% faster load times |
| [**EliteNotes App**](https://elite-notes-poc.vercel.app/) | React.js, Firebase, NLP | 7 ML-powered features (transcription, tagging, summarization) |
| **3D PCA Visualization** | Python, Plotly, Word2Vec | Visualized 10K+ word embeddings, reducing complexity by 85% |
| **Apparel Recommender** | Python, TF-IDF, CNN, Word2Vec | Improved recommendation accuracy by 20% with text & image embeddings |

### CIFAR-10 CNN Details
- **Architecture**: 9 convolutional layers with batch normalization, max pooling, dropout, and 3 fully connected layers
- **Dataset**: CIFAR-10 (60,000 32x32 RGB images, 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training**: 100 epochs, SGD (momentum=0.9, weight decay=1e-6), CrossEntropyLoss, LR scheduling (initial 0.01, reduced on plateau)
- **Data Augmentation**: Color jitter, random perspective, random horizontal flip, normalization
- **Usage**:
  ```python
  from huggingface_hub import from_pretrained_pytorch
  model = from_pretrained_pytorch("chandu1617/CIFAR10-CNN_Model")
