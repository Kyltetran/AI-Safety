# AI Safety in Fake News Detection: A Comprehensive Dataset Analysis

## Overview

This repository documents our research journey exploring **AI safety considerations in fake news detection**. We systematically evaluated **five major datasets** using both classical machine learning and deep learning approaches to understand their strengths, limitations, and real-world applicability.

> **Important:** This is a research and educational project. We also built an **educational website** to share our findings—not a detection tool or platform. Our goal is to inform and educate about the challenges, methodologies, and safety concerns in AI-powered misinformation detection.

---

## Research Motivation

Fake news detection isn't just a classification problem—it's a **safety-critical AI application** where mistakes can:
- Suppress legitimate information
- Amplify harmful misinformation
- Disproportionately impact marginalized communities
- Erode public trust in information systems

We prioritize **interpretability, robustness, and fairness** over raw accuracy metrics.

---

## Datasets Evaluated

### 1. Real & Fake News Dataset
**Size:** 44,000+ articles (21k real, 23k fake)

**Limitations:**
- **Severe data leakage** - Models achieved 99%+ accuracy (too good to be true)
- **Train/test contamination** - Duplicates not properly removed
- **Superficial learning** - Models learning word artifacts rather than semantic meaning

**Performance:** ML models (98.9-99.7% accuracy), LSTM (99.7% at epoch 20)

**Conclusion:** Unreliable for research. High accuracy is misleading and won't generalize to real-world scenarios.

### 2. COVID-19 Fake News Dataset
**Size:** 10,000+ rows (6k train, 2k validation, 2k unlabeled test)

**Limitations:**
- **Domain-specific** - Limited to COVID-19 context only
- **Too small** - Insufficient for robust BERT/transformer fine-tuning
- **No labeled test set** - Cannot properly evaluate generalization
- **Prone to overfitting** - High validation accuracy but questionable real-world performance
- **Post-pandemic irrelevance** - Limited practical utility now

**Best Performance:** RoBERTa with focal loss + class weighting (98.1% validation accuracy, 0.03 loss)

**Other Models Tested:** BERT, DistilBERT, DeBERTa (~85% when combined with other datasets)

**Conclusion:** Good for COVID-specific analysis but lacks real-world applicability beyond the pandemic context.


### 3. BuzzFeed Political News Dataset
**Size:** 182 articles (91 real, 91 fake)

**Limitations:**
- **Critically undersized** - Only 182 samples total
- **Cannot train deep learning** - Insufficient data
- **Poor generalization** - DL models only reach ~62% test accuracy

**Performance:** Best result was 85% using Passive Aggressive Classifier without preprocessing

**Conclusion:** Too small for meaningful model training. Suitable only as a qualitative benchmark.



### 4. PHEME Rumor Detection Dataset
**Size:** 60,000+ tweets

**Limitations:**
- **Severe class imbalance** - 42k rumors vs 6k non-rumors (7:1 ratio)
- **Computationally intensive** - DeBERTa crashes on Kaggle, out of memory on Colab
- **Requires local GPU** - Needs 16GB+ RAM for training

**Performance Comparison:**
- **DeBERTa (DL):** 96.4% validation accuracy, 0.23 loss - but requires significant compute
- **Logistic Regression (ML):** 88% accuracy
- **SVM (ML):** 82% accuracy

**Conclusion:** Shows promise but requires advanced imbalance handling techniques and substantial computational resources.


### 5. WELFake Dataset **[SELECTED AS PRIMARY]**
**Size:** 72,000+ articles (35k real, 37k fake)

**Why WELFake Stands Out:**
- **Perfectly balanced** - Near 50/50 split between real and fake
- **Hybrid features** - Combines word embeddings with linguistic features (POS tags, readability, sentiment)
- **Diverse sources** - Multiple domains and writing styles
- **Well-documented** - Strong data provenance and preprocessing standards
- **Ethically designed** - Explicitly addresses fairness and transparency

**Classical ML Performance (TF-IDF):**
- Random Forest: **96.5%**
- Logistic Regression: **96.5%**
- Linear SVM: **96.5%**
- AdaBoost: 95.0%
- XGBoost: 94.1%
- Decision Tree: 94.9%
- Gaussian Naive Bayes: 85.0%
- KNN: 77.0%

**Deep Learning Performance:**
- CNN-LSTM (GloVe): **98.21%** (val_loss: 0.0534, 20 epochs) **Best**
- BERT-CNN: **98.15%** (val_loss: 0.1190, 20 epochs)
- BERT-CNN-BiLSTM: **98.13%** (val_loss: 0.0809, 20 epochs)
- CNN-PCA (GloVe): 98.05% (val_loss: 0.0538, 20 epochs)
- CNN (GloVe): 97.14% (val_loss: 0.2470, 40 epochs)

**Computational Requirements:**
- Dataset size: ~200MB
- GloVe embeddings: ~300MB
- Training time: 20-30 minutes for ML, 1-2 hours for CNN-LSTM, 3-5 hours for BERT models (GPU required)

**Conclusion:** Best choice for robust fake news detection research. Balanced, interpretable, and comprehensive.

---

## Key Research Findings

### Classical ML vs. Deep Learning Trade-offs

**Classical ML (Random Forest, Logistic Regression, SVM):**
#### **Pros**
- Fast training (minutes)
- Highly interpretable
- Competitive performance (96.5% on WELFake)

#### **Cons**
- Lower computational requirements
- May miss complex semantic patterns

**Deep Learning (CNN-LSTM, BERT):**
#### **Pros**
- Slightly better accuracy (98%+ on WELFake)
- Captures semantic nuances

#### **Cons**
- Computationally expensive (hours of GPU training)
- Less interpretable (black-box)
- Requires large datasets

**Key Insight:** For WELFake, classical ML methods achieve 96.5% accuracy in minutes, while deep learning gains only ~2% more accuracy but requires hours of GPU training. The trade-off depends on deployment context.

---

## Interpretability Analysis with SHAP

We applied **SHAP (SHapley Additive exPlanations)** to understand model decision-making:

### Logistic Regression Insights:
- **Real news indicators:** "taxes," "capitol," "war," "government"
- **Fake news indicators:** "terror," "suspect," "senator," "stop"
- **Interpretation:** Linear, transparent feature weights—easy to audit

### XGBoost Insights:
- **Top features:** "surveillance," "middle," "weapons"
- **Behavior:** Context-dependent, captures non-linear feature interactions
- **Method:** TreeSHAP provides exact feature attributions

### Random Forest Limitation:
- Distributed decision-making across many trees
- SHAP analysis computationally expensive
- Less interpretable than linear models despite high performance

**Key Insight:** Linear models (Logistic Regression, SVM) provide the clearest explanations. XGBoost balances performance with interpretability. BERT models remain largely opaque despite highest accuracy.

---

## Lessons Learned

### 1. High Accuracy ≠ Good Model
The Real & Fake News dataset achieved 99%+ accuracy but suffered from data leakage. This teaches us to be skeptical of perfect results and emphasizes **rigorous data validation**.

### 2. Dataset Quality > Model Complexity
WELFake's careful construction (balanced, diverse, well-documented) enabled both classical ML and DL to perform well. A clean dataset matters more than sophisticated architecture.

### 3. Interpretability Has Real Value
SHAP analysis revealed that models rely on reasonable linguistic patterns, not spurious correlations. This builds trust and enables debugging.

### 4. Context Matters for Deployment
COVID-19 dataset had good metrics but limited real-world utility post-pandemic. Models must match deployment context.

### 5. Trade-offs Are Unavoidable
No single model excels at accuracy, speed, interpretability, and robustness simultaneously. Deployment decisions require explicit trade-off analysis.

---

## Educational Mission

This research supports an **educational website** that:
- Explains fake news detection challenges to non-technical audiences
- Demonstrates trade-offs between different approaches
- Highlights AI safety considerations often overlooked in industry
- Provides transparent documentation of our research process

**We are researchers and educators, not building a commercial detection tool.** Our goal is to inform better decision-making about AI deployment in high-stakes information environments.
