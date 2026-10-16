 <div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:090909,60:171717,100:541515&height=210&section=header&text=CHANDRA%20PRAKASH%20BATHULA&fontSize=34&fontColor=ffffff&fontAlignY=40&desc=Machine%20Learning%20%7C%20Computer%20Vision%20%7C%20Deep%20Learning&descAlignY=59&descSize=16" width="100%" alt="Chandra Prakash Bathula — Machine Learning Research and Engineering"/>
  <br/>
  <br/>
  <a href="https://www.linkedin.com/in/chandra-prakash-bathula">
    <img src="https://img.shields.io/badge/LinkedIn-Professional%20Profile-333333?style=flat-square&logo=linkedin" alt="LinkedIn"/>
  </a>
  <a href="https://github.com/ChandraPrakash-Bathula">
    <img src="https://img.shields.io/badge/GitHub-Research%20%26%20Code-333333?style=flat-square&logo=github" alt="GitHub"/>
  </a>
  <a href="https://huggingface.co/chandu1617">
    <img src="https://img.shields.io/badge/Hugging%20Face-Models%20%26%20Demos-333333?style=flat-square&logo=huggingface" alt="Hugging Face"/>
  </a>
  <a href="https://medium.com/@ChandraPrakash-Bathula">
    <img src="https://img.shields.io/badge/Medium-Technical%20Writing-333333?style=flat-square&logo=medium" alt="Medium"/>
  </a>
  <br/>
  <br/>
  <img src="https://komarev.com/ghpvc/?username=ChandraPrakash-Bathula&style=flat-square&color=7d2020&label=PROFILE+VIEWS" alt="Profile Views"/>
</div>

---

## About

I am a machine learning engineer, researcher, and adjunct faculty member at Saint Louis University, working at the intersection of deep learning, computer vision, efficient model adaptation, and AI systems.

My work combines controlled experimentation, empirical evaluation, model development, and software engineering. I am interested in understanding not only how well models perform, but also **why they perform differently, which experimental factors drive the results, and what conclusions the evidence supports**.

I also develop interactive research and educational tools that make machine learning systems easier to inspect, evaluate, and reproduce.

- **Research:** Deep learning evaluation, computer vision, neural architectures, and efficient inference.
- **Methods:** Controlled benchmarks, ablation studies, empirical analysis, and reproducible experimentation.
- **Engineering:** PyTorch, Hugging Face, model-serving infrastructure, and interactive ML applications.
- **Teaching:** Applied machine learning, deep learning, NLP, retrieval-augmented generation, and AI agents.

## Research Interests

<div align="center">

| Computer Vision | Efficient Deep Learning | ML Evaluation |
|:---:|:---:|:---:|
| CNNs and Transformers | LoRA and QLoRA | Controlled benchmarking |
| Visual representations | Speculative decoding | Reproducibility |
| Hybrid architectures | Test-time compute | Experimental validity |

</div>

Additional interests include multimodal learning, retrieval-augmented generation, agentic systems, and machine learning for network security.

---

## Selected Publications

### NeurIPS 2026

**Separating Architecture from Initialization: What a Controlled CNN-Transformer Benchmark Can and Cannot Attribute**

NeurIPS 2026 Workshop: Can We Trust AI Evaluation? Robustness, Causality, and Risk in Modern AI Assessment.

*First-author work on controlled comparisons of CNNs and Transformers across architectures, datasets, and training regimes.*

**What Fine-Tuning Costs: A Measured Comparison of Full Fine-Tuning, LoRA, and QLoRA**

NeurIPS 2026 Education Track · Co-authored.

*An empirical comparison of adaptation methods, trainable parameters, memory consumption, checkpoint size, and execution accuracy.*

**Same Compute, Different Allocation: An Interactive Learning Experience for Test-Time Compute**

NeurIPS 2026 Education Track · Co-authored.

*An interactive exploration of test-time compute allocation and its performance trade-offs.*

**Fewer Passes, Same Output: Exploring the Performance Trade-offs of Speculative Decoding**

NeurIPS 2026 Education Track · First author.

*An empirical investigation of speculative decoding, verification overhead, and inference-speed trade-offs.*

### Previous Publication

**Developing an Enterprise Application Tool to Discover Midwest Job Trends**

*Journal of Computing Sciences in Colleges*, 40(6), 75–89, 2025. ACM.

[DOI](https://doi.org/10.5555/3729857.3729866)

---

## Research & Selected Projects

### Controlled CNN–Transformer Benchmark

**Focus:** Architecture evaluation, initialization effects, and experimental validity.

- Compared 10 vision architectures across nine datasets and three training regimes.
- Investigated the effects of training from scratch versus using pretrained representations.
- Designed controlled experiments to distinguish architecture-related effects from initialization-related effects.
- Examined dataset-dependent performance differences and limitations in cross-architecture comparisons.

**Technologies:** PyTorch, CNNs, Vision Transformers, Swin Transformers, experimental benchmarking.

### LLM Fine-Tuning Laboratory

**Focus:** Parameter-efficient adaptation and computational cost.

- Evaluated 11 variants of Qwen2.5-7B-Instruct on WikiSQL using full fine-tuning, LoRA, and QLoRA.
- Measured execution accuracy, GPU memory consumption, training time, and checkpoint size.
- Released model variants with configurations, environment files, evaluation metrics, and checkpoint validation.
- Documented single-seed limitations and the need for additional data-scale experiments.

**Technologies:** PyTorch, Hugging Face Transformers, PEFT, bitsandbytes, SQLite.

### Speculative Decoding

**Focus:** Efficient language-model inference.

- Investigated speculative decoding using smaller draft and larger target models.
- Measured target-model verification requirements and end-to-end speedup.
- Developed an interactive demonstration of the trade-offs between acceptance rate, speculation length, and inference efficiency.

**Technologies:** Transformers, Python, Hugging Face, interactive ML applications.

### Agentic AI Benchmarking

**Focus:** Tool use, task completion, and comparative evaluation.

- Evaluated no-code, low-code, and LLM-based agents across 36 tasks.
- Examined reasoning, academic workflows, and tool-calling behavior under multiple prompting strategies.
- Built an evaluation framework for comparing agent approaches across task categories.

### ML for Network Security

**Focus:** Reproducible anomaly-detection evaluation.

- Developed benchmarking infrastructure for 20+ machine learning models across multiple cyberattack categories and datasets.
- Worked on consistent evaluation pipelines and comparative model analysis.
- Investigated the use of machine learning to identify suspicious patterns in network traffic.

---

## Research Software & Interactive Demonstrations

I build tools that connect model development with measurable experiments and interactive exploration.

| Project | Description | Resources |
|---|---|---|
| WikiSQL Fine-Tuning Laboratory | Compare model adaptation methods, memory use, training cost, and execution accuracy. | [Hugging Face](https://huggingface.co/chandu1617) |
| CIFAR-10 CNN | Convolutional neural network trained for image classification. | [Model](https://huggingface.co/chandu1617/CIFAR10-CNN_Model) · [Live Demo](https://huggingface.co/spaces/chandu1617/cifar10-cnn-app) |
| Food-101 Classification | ResNet50-based food image classification. | [Model](https://huggingface.co/chandu1617/Food_101_Classification_ResNet50) |
| NYC Taxi Duration Prediction | Interactive machine learning application for trip-duration prediction. | [Live Demo](https://huggingface.co/spaces/chandu1617/NYC_Taxi_Trip_Duration_Predictor) |
| Show the Wire | Interactive exploration of LLM tool discovery and calling. | [Live Demo](https://huggingface.co/spaces/ChandraPrakashBathula/show-the-wire) |

Explore additional implementations, experiments, and repositories through my [GitHub profile](https://github.com/ChandraPrakash-Bathula) and [Hugging Face profile](https://huggingface.co/chandu1617).

---

## Teaching & Academic Experience

**Adjunct Faculty — Saint Louis University, School for Professional Studies**

*August 2024 – Present*

- Designed and taught applied deep learning material covering PyTorch, CNNs, Transformers, NLP, RAG, and AI agents.
- Developed more than 10 interactive machine learning playgrounds to support practical experimentation.
- Taught more than 300 graduate students across applied computing and analytics courses.
- Mentored 20+ capstone and research projects, covering problem formulation, implementation, experimentation, and technical communication.

My teaching and software development work focuses on making complex machine learning concepts understandable through practical implementation, visual analysis, and hands-on experimentation.

---

## Technical Skills

<div align="center">
  <img src="https://skillicons.dev/icons?i=python,pytorch,tensorflow,sklearn,fastapi,docker,aws,react,typescript,postgres,mongodb,git&perline=6" alt="Python, PyTorch, TensorFlow, scikit-learn, FastAPI, Docker, AWS, React, TypeScript, PostgreSQL, MongoDB, and Git"/>
</div>

**Machine Learning:** PyTorch, TensorFlow, scikit-learn, CNNs, Transformers, model evaluation, experimental design.

**Efficient AI:** LoRA, QLoRA, parameter-efficient fine-tuning, speculative decoding, inference benchmarking.

**Generative AI:** Hugging Face, RAG, FAISS, LangChain, embeddings, LLM APIs, agentic workflows.

**Software & Infrastructure:** Python, SQL, TypeScript, JavaScript, FastAPI, React, Docker, AWS, PostgreSQL, MongoDB, Git, CI/CD.

---

## Education

**Saint Louis University** · St. Louis, Missouri

**M.S. in Information Systems** · May 2024

- GPA: 3.93/4.00.
- Distinguished Graduate with Distinction for the master's research project and overall academic performance.

---

## Recognition & Academic Service

- First-author publication at the CCSC Central Plains Conference proceedings.
- Research presentation and poster judging at the CCSC Central Plains Conference, Drake University, 2025.
- Distinguished Graduate recognition at Saint Louis University.
- Mentorship of 20+ capstone and research projects.

---

## Connect

<div align="center">
  <a href="https://portfolio-chandra-prakash-bathulas-projects.vercel.app">
    <img src="https://img.shields.io/badge/Portfolio-Research%20%26%20Projects-333333?style=for-the-badge&logo=firefox" alt="Portfolio"/>
  </a>
  <a href="https://github.com/ChandraPrakash-Bathula">
    <img src="https://img.shields.io/badge/GitHub-Code%20%26%20Experiments-333333?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  <a href="https://huggingface.co/chandu1617">
    <img src="https://img.shields.io/badge/Hugging%20Face-Models%20%26%20Demos-333333?style=for-the-badge&logo=huggingface" alt="Hugging Face"/>
  </a>
  <a href="https://www.linkedin.com/in/chandra-prakash-bathula">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-333333?style=for-the-badge&logo=linkedin" alt="LinkedIn"/>
  </a>
  <a href="mailto:chandraprakash.bathula@slu.edu">
    <img src="https://img.shields.io/badge/Email-Contact-333333?style=for-the-badge&logo=gmail" alt="Email"/>
  </a>
</div>

<div align="center">
  <br/>
  <sub>Building, evaluating, and understanding machine learning systems through rigorous experimentation.</sub>
  <br/>
  <br/>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:541515,60:171717,100:090909&height=100&section=footer" width="100%" alt="Footer"/>
</div>
