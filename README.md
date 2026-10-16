# Chandra Prakash Bathula

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:101820,60:183A4A,100:2C5364&height=200&section=header&text=CHANDRA%20PRAKASH%20BATHULA&fontSize=34&fontColor=FFFFFF&fontAlignY=40&desc=Machine%20Learning%20%7C%20Computer%20Vision%20%7C%20AI%20Systems&descAlignY=60&descSize=16" width="100%" alt="Chandra Prakash Bathula — Machine Learning Research and Engineering"/>

<img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=500&size=17&pause=1500&color=2C5364&center=true&vCenter=true&width=750&lines=Deep+Learning+Research+%26+Evaluation;Computer+Vision+%26+Neural+Architectures;Efficient+LLM+Adaptation+%26+Inference;Reproducible+Experiments+%26+Interactive+ML+Systems" alt="Research interests"/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/chandra-prakash-bathula)
[![GitHub](https://img.shields.io/badge/GitHub-Repositories-181717?style=flat-square&logo=github)](https://github.com/ChandraPrakash-Bathula)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Models_%26_Demos-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/chandu1617)
[![Portfolio](https://img.shields.io/badge/Portfolio-Website-2C5364?style=flat-square&logo=firefox)](https://portfolio-chandra-prakash-bathulas-projects.vercel.app)
[![Email](https://img.shields.io/badge/Email-Contact-64748B?style=flat-square&logo=gmail&logoColor=white)](mailto:chandraprakash.bathula@slu.edu)

</div>

---

## About

I am an AI/ML researcher, engineer, and Adjunct Faculty member at Saint Louis University, with an M.S. in Information Systems (GPA: 3.93/4.00).

My work focuses on understanding, evaluating, and building deep learning systems through controlled experiments, reproducible benchmarks, and practical implementations. My research interests include computer vision, neural architecture evaluation, efficient large language model adaptation and inference, and machine learning systems.

I also develop interactive tools that make machine learning methods observable and experimentally accessible, connecting model implementation with empirical analysis.

## Research Interests

- **Computer Vision:** CNNs, Vision Transformers, hybrid architectures, visual representation learning, and cross-dataset evaluation.
- **Deep Learning Evaluation:** Controlled benchmarking, initialization effects, experimental design, reproducibility, and interpretation of model performance.
- **Efficient LLMs:** Full fine-tuning, LoRA, QLoRA, speculative decoding, and test-time compute allocation.
- **Retrieval and Agentic Systems:** Retrieval-augmented generation, tool use, agent evaluation, and interactive AI systems.
- **ML for Security:** Machine learning benchmarks for network anomaly detection and cyberattack classification.
- **AI Education and Research Software:** Interactive experiments, visualization, and reproducible learning environments.

## Publications

### NeurIPS 2026 Workshop

**[Separating Architecture from Initialization: What a Controlled CNN-Transformer Benchmark Can and Cannot Attribute](#)**

*Can We Trust AI Evaluation? Robustness, Causality, and Risk in Modern AI Assessment — NeurIPS 2026 Workshop.*

- **Author:** Chandra Prakash Bathula
- **Focus:** Controlled comparison of convolutional and attention-based vision architectures across datasets and training regimes, examining the role of pretraining and initialization in measured performance.

### NeurIPS 2026 Education Track

**What Fine-Tuning Costs: A Measured Comparison of Full Fine-Tuning, LoRA, and QLoRA**

- **Authors:** Chandra Prakash Bathula, Ram Mohan Bathula
- **Focus:** Empirical comparison of fine-tuning methods for parameter efficiency, GPU memory, checkpoint size, training cost, and execution accuracy.

**Same Compute, Different Allocation: An Interactive Learning Experience for Test-Time Compute**

- **Authors:** Chandra Prakash Bathula, Ram Mohan Bathula
- **Focus:** Interactive exploration of test-time compute allocation and its effect on model performance and computational trade-offs.

**Fewer Passes, Same Output: Exploring the Performance Trade-offs of Speculative Decoding**

- **Author:** Chandra Prakash Bathula
- **Focus:** Experimental analysis of speculative decoding, including target-model evaluations, verification costs, and practical inference speedups.

### Journal / Conference Publications

**Developing an Enterprise Application Tool to Discover Midwest Job Trends**

- **Authors:** Chandra Prakash Bathula & Maria Weber.
- **Venue:** Journal of Computing Sciences in Colleges (CCSC), ACM, 40(6), 75–89, 2025.
- **DOI:** [10.5555/3729857.3729866](https://doi.org/10.5555/3729857.3729866)

> Publication status and bibliographic details should match the official proceedings. Add links to the final workshop and Education Track papers when their official proceedings pages become available.

## Research Experience

### Controlled Evaluation of CNNs and Transformers

- Compared 10 vision architectures across nine datasets and three training regimes to investigate the effects of architecture, initialization, and pretraining.
- Examined differences between training from scratch and fine-tuning pretrained models.
- Developed a controlled benchmarking workflow to support reproducible comparisons and more careful interpretation of architectural claims.

### Efficient LLM Fine-Tuning

- Evaluated 11 variants of Qwen2.5-7B-Instruct using full fine-tuning, LoRA, and QLoRA on WikiSQL.
- Measured execution accuracy, trainable parameters, GPU memory, training time, and checkpoint size.
- Implemented evaluation using held-out examples and SQLite execution, with experiment configurations and checkpoint validation.
- Investigated rank-dependent behavior and documented limitations associated with single-seed experiments and dataset size.

### Speculative Decoding and Test-Time Compute

- Evaluated speculative decoding trade-offs using smaller draft models and larger target models.
- Investigated target-model verification frequency, computational overhead, and measured inference speedup.
- Built interactive tools for examining test-time compute allocation.

### Network Anomaly Detection

- Developed benchmarking infrastructure for evaluating 20+ machine learning models across multiple datasets and eight or more cyberattack categories.
- Focused on systematic model comparison and reproducible evaluation workflows.

## Selected Research and Engineering Projects

| Project | Description | Technologies |
|---|---|---|
| [WikiSQL Fine-Tuning Laboratory](https://huggingface.co/chandu1617) | Fine-tuning experiments and evaluation of full fine-tuning, LoRA, and QLoRA | PyTorch, Transformers, PEFT, bitsandbytes, SQLite |
| [CIFAR-10 CNN](https://huggingface.co/chandu1617/CIFAR10-CNN_Model) | Nine-layer convolutional neural network with an interactive inference demo | PyTorch, CNNs, Hugging Face |
| [CIFAR-10 CNN Demo](https://huggingface.co/spaces/chandu1617/cifar10-cnn-app) | Interactive image classification | PyTorch, Hugging Face Spaces |
| [Food-101 Classification](https://huggingface.co/chandu1617/Food_101_Classification_ResNet50) | ResNet-50-based, 101-class image classification | PyTorch, ResNet, Computer Vision |
| [Show the Wire](https://huggingface.co/spaces/ChandraPrakashBathula/show-the-wire) | Interactive demonstration of LLM tool discovery and calling | LLMs, Tool Calling, Hugging Face |
| [NYC Taxi Trip Duration Predictor](https://huggingface.co/spaces/chandu1617/NYC_Taxi_Trip_Duration_Predictor) | Deployed machine learning prediction application | Python, Machine Learning, Hugging Face Spaces |

## Teaching and Mentoring

**Adjunct Faculty — Saint Louis University, School for Professional Studies**  
*August 2024 – Present*

- Designed and taught applied deep learning content covering PyTorch, CNNs, Transformers, NLP, RAG, and AI agents.
- Developed 10+ interactive ML playgrounds for hands-on experimentation.
- Taught and supported 300+ graduate students across software development, information systems, and analytics courses.
- Mentored 20+ capstone and research projects across problem formulation, implementation, evaluation, and technical writing.
- Built RAG Pipeline Studio using FastAPI, React, FAISS, Ollama, and hosted model APIs to compare retrieval configurations and model outputs.

## Technical Skills

**Programming:** Python, SQL, R, JavaScript, TypeScript

**Machine Learning:** PyTorch, TensorFlow, scikit-learn, CNNs, Transformers, model evaluation, experimental design

**LLMs and Generative AI:** Hugging Face, PEFT, LoRA, QLoRA, speculative decoding, RAG, FAISS, LangChain, agentic workflows

**Computer Vision:** Convolutional architectures, ResNet, Vision Transformers, image classification, architecture benchmarking

**Backend and Deployment:** FastAPI, React, Docker, AWS, PostgreSQL, MongoDB, CI/CD

**Visualization and Evaluation:** Plotly, UMAP, metrics analysis, reproducible experiments, interactive ML applications

## Education

**Saint Louis University** — St. Louis, Missouri, USA

**Master of Science in Information Systems** | May 2024

- GPA: 3.93/4.00
- Distinguished Graduate
- Distinction for master's research project

## Academic and Professional Activities

- Paper presentation and poster judging at the CCSC Central Plains Conference, Drake University, 2025.
- Development of interactive machine learning education resources and reproducible research software.
- Mentoring graduate students and supporting applied research and capstone projects.

---

<div align="center">

### Connect

[Website](https://portfolio-chandra-prakash-bathulas-projects.vercel.app) · [LinkedIn](https://www.linkedin.com/in/chandra-prakash-bathula) · [GitHub](https://github.com/ChandraPrakash-Bathula) · [Hugging Face](https://huggingface.co/chandu1617) · [Medium](https://medium.com/@ChandraPrakash-Bathula)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2C5364,60:183A4A,100:101820&height=100&section=footer" width="100%" alt="Footer"/>

</div>
