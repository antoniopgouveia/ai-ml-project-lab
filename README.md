🤖 AI/ML Project Lab

Applied AI/ML portfolio by Antonio Gouveia — building solutions to real-world problems within business intelligence and biotechnology using regression, classification, and deep learning architectures.

👋 About
I'm an AI/ML and Business Intelligence professional based in Porto, Portugal, passionate about applying machine learning to problems where data meets decision-making. My work sits at the intersection of business intelligence and biotechnology — areas where well-designed AI systems can create measurable real-world impact.

This repository is a living collection of end-to-end ML projects. Each one is framed around a genuine domain question, built with reproducible code, and evaluated against meaningful baselines — not just notebooks that run once and are forgotten.

🔭 Currently working on: Chest X-Ray Disease Classifier and Clinical Note Summarization System

🌱 Exploring: Medical imaging, biomedical NLP, clinical AI, retrieval-augmented generation

💬 Ask me about: NLP, neural networks, business intelligence, Python, Azure

📫 Contact: LinkedIn · GitHub · antoniopagouveia@gmail.com

🧠 Methods & Techniques
This portfolio covers a progression from foundational machine learning to modern deep learning and LLM engineering:

Regression & Classification
Traditional supervised learning using linear/logistic regression, decision trees, and ensemble methods (XGBoost, Random Forest). Applied to tabular data problems in business intelligence, clinical trial outcome prediction, and sports analytics — with model interpretability via SHAP.

Artificial Neural Networks (ANNs)
Feedforward neural networks for structured data problems — from predictive modeling to feature representation learning. Built with PyTorch, with experiment tracking via Weights & Biases and MLflow.

Convolutional Neural Networks (CNNs)
Image classification and medical imaging applications — multi-label disease detection from chest X-rays, model interpretability with Grad-CAM attention maps, and transfer learning with pre-trained architectures (DenseNet, ResNet, EfficientNet).

Recurrent Neural Networks (RNNs)
Sequential data modeling for time-series forecasting, anomaly detection, and sequence-to-sequence tasks. Covering LSTMs, GRUs, and encoder-decoder architectures for temporal patterns in business metrics and clinical text.

Natural Language Processing (NLP)
Text classification, named entity recognition, relation extraction, and biomedical NLP using transformer models (BERT, PubMedBERT, BioLinkBERT). Working with domain-specific language in medical and clinical contexts.

Retrieval-Augmented Generation (RAG)
LLM engineering with retrieval pipelines — embedding-based search, vector stores (ChromaDB, FAISS), generation with citations, and evaluation frameworks (RAGAS, ROUGE, BERTScore). Deployed as interactive applications via Streamlit and Gradio.

🛠️ Tech Stack
Category	Tools & Frameworks
Languages	Python, SQL
Classical ML	scikit-learn, XGBoost, LightGBM
Deep Learning	PyTorch, torchvision
NLP / Transformers	HuggingFace Transformers, LangChain, sentence-transformers
RAG	ChromaDB, FAISS, RAGAS
Data	pandas, NumPy, Polars
Visualization	Matplotlib, Seaborn, Plotly, SHAP, Grad-CAM
Deployment	Streamlit, Gradio, FastAPI, Docker, HuggingFace Spaces
Experiment Tracking	Weights & Biases, MLflow
Cloud / Infra	Microsoft Azure, GitHub Actions CI/CD
Dev Tools	VS Code, Google Colab, Jupyter, Git

📁 Repository Structure
Each project is self-contained in its own folder with a consistent layout:

text
ai-ml-project-lab/
├── README.md                              ← You are here
├── LICENSE                                ← MIT License
├── .gitignore
│
├── chest-xray-classifier/
│   ├── README.md                          ← Problem, approach, results, how to run
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_model_training.ipynb
│   │   └── 03_evaluation_gradcam.ipynb
│   ├── src/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── gradcam.py
│   ├── app/
│   │   └── gradio_app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── clinical-note-summarizer/
│   ├── README.md
│   ├── notebooks/
│   ├── src/
│   ├── requirements.txt
│   └── ...
│
└── ... (future projects)
Per-Project Standards
Element	Purpose
README.md	Problem statement, approach, results, how to run, demo link
notebooks/	Numbered EDA → modeling → evaluation workflow
src/	Modular, importable Python code (not just notebooks)
requirements.txt	Pinned dependencies for full reproducibility
app/	Deployment code (Streamlit / Gradio / FastAPI) where applicable
🔬 Approach & Philosophy
Every project in this repository follows the same principles:

Problem first — Each project starts with a real question, not a technique in search of a dataset

Reproducible by default — Clone, install, run — no hidden steps or undocumented dependencies

Honest evaluation — Models are compared against baselines with real metrics (accuracy, F1, ROUGE, AUROC, latency), including failure analysis

Deployment-ready — Where feasible, projects include a live demo deployed to HuggingFace Spaces or Streamlit Community Cloud

Documented decisions — Notebooks explain why a method was chosen, not just what was used

📫 Connect
LinkedIn: linkedin.com/in/YOUR_LINKEDIN

GitHub: github.com/YOUR_GITHUB

HuggingFace: huggingface.co/YOUR_HF_USERNAME

Email: antoniopagouveia@gmail.com
