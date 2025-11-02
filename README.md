
<br>
 
 
 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]


<br>

# <p align="center"> 5- Social [Buzz AI]() - Natural Language Processing (NLP) - [Class 2]() - Lesson on Vector Representations and Text Classification



<br><br>


<p align="center">
   <img src="https://github.com/user-attachments/assets/791a69e2-d09a-429f-9257-f6667fff5c04 ">
 </p>

<br><br>

[**Course:**]() Humanistic AI & Data Science (4th Semester)  
[**Institution:**]() PUC-SP  
**Professor:**  [✨ Rooney Ribeiro Albuquerque Coelho](https://www.linkedin.com/in/rooney-coelho-320857182/)



<br><br>


#### <p align="center"> [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-%C2%B7%C2%B7%C2%B7%20Mindful%20AI%20Assistants%20%C2%B7%C2%B7%C2%B7-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants)


<br><br>


> [!TIP]
>
>  This repository 2-social-buzz-ai-GBoost-and-LowDefault-Modeling is part of the main project 1-social-buzz-ai-main.
>  To explore all related materials, analyses, and notebooks, visit the main repository 
>
> * [1-social-buzz-ai-main](https://github.com/Mindful-AI-Assistants/1-social-buzz-ai-main)
> *Part of the Humanistic AI Research & Data Modeling Series — where data meets human insight.*
>
> * [4- Social Buss: NLP - Class 1](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NL-Class_1) 
> 
> * [Embedding Projector](https://projector.tensorflow.org/)
> 
>


<!--Confidentiality Statement-->


<br><br>


> [!IMPORTANT]
>
> ⚠️ Heads Up 
>
> * Projects and deliverables may be made [publicly available]() whenever possible.
>
> * The course prioritizes [**hands-on practice**]() with real data in consulting scenarios.
>
> *  All activities comply with the [**academic and ethical guidelines of PUC-SP**]().
>
> * [**Confidential information**]() from this repository remains private in [private repositories]().
>
>

#  

<br><br><br>

<!--End-->

> [!TIP]
>
> * [Access](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NL-Class_1/blob/7a5b1e2ad8bee693c6842a3b79a38c3d7d668239/1-Workbook_Natural%20Language%20Processing%20-%20Class%201.pdf)  Workbook - (Class 1 and 2)
> 
> * [Access](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NL-Class_1/blob/1170f52a88b432225e216b3519810285d65b3066/1_%F0%9F%87%AC%F0%9F%87%A7_NLP_Pre_Processing_ENGLISH.ipynb):  🇬🇧 1- NLP_Pre_Processing_ENGLISH
>
> * [Access](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NL-Class_1/blob/f395b6b6ffba24b65dd6e593e4bb3b3b899301e0/1_%F0%9F%87%A7%F0%9F%87%B7NLP_PreProcessing_Portuguese.ipynb):   🇧🇷 1-Code NLP_Pre_Processing_Portuguese
>
>
>  * [Access](https://github.com/Mindful-AI-Assistants/4-social-buzz-ai--Natural_Language_Processing-NL-Class_1):  NLP  - Class 1 Repo
>
 

<br><br><br>



<!--

## [Libraries Used]()


<br>

- `numpy`
- `pandas`
- `nltk`
- `sklearn`
- `gensim`
- `safetensors`
- `transformers`
- `pickle`

-->



## Table of Contents

1. [What Is This Notebook About?](#what-is-this-notebook-about)
2. [Playing With Numbers (Vectors)](#playing-with-numbers-vectors)
3. [Finding Out If Two Things Are Alike (Cosine Similarity)](#finding-out-if-two-things-are-alike-cosine-similarity)
4. [Using Secret Codes For Words (Word Embeddings)](#using-secret-codes-for-words-word-embeddings)
5. [Super-Secret Codes: Transformers and BERT](#super-secret-codes-transformers-and-bert)
6. [Getting Our Messages (Loading Data)](#getting-our-messages-loading-data)
7. [Cleaning Up Our Messages (Text Preprocessing)](#cleaning-up-our-messages-text-preprocessing)
8. [Turning Words Into Numbers (Bag of Words)](#turning-words-into-numbers-bag-of-words)
9. [Splitting Our Messages (Training and Testing)](#splitting-our-messages-training-and-testing)
10. [Teaching Our Computer To Guess Feelings (Model Training)](#teaching-our-computer-to-guess-feelings-model-training)
11. [Saving Our Computer’s Brain For Later (Model Persistence)](#saving-our-computers-brain-for-later-model-persistence)
12. [Asking The Computer To Guess For Us! (Interactive Prediction)](#asking-the-computer-to-guess-for-us-interactive-prediction)


<br><br>



## 1. What Is This Notebook About?

This notebook is like a **recipe book** for teaching computers how to understand what people write or say. It shows step-by-step how to take words, turn them into numbers, teach a computer about them, and then let it guess if a message is, for example, happy or sad!


<br><br>

## 2. Playing With Numbers (Vectors)

<br>

**Why?** Numbers help computers see words!
**What is it?** We make some number sequences called *vectors*. Then, we add, subtract, multiply and compare them. This is the building block for more magic later.
**Code:**

<br>

```python
import numpy as np
vetor1 = np.array([1., 2., 1., 4.])
vetor2 = np.zeros(4)
vetor3 = np.ones(4)
print("Vetor 1", vetor1)
print("Vetor 2", vetor2)
print("Vetor 3", vetor3)
print("Add", vetor1 + vetor3)
print("Subtract", vetor1 - vetor3)
print("Multiply", vetor1 * vetor3)
print("Dot Product", np.dot(vetor1, vetor3))
```


<br><br>

## 3. Finding Out If Two Things Are Alike (Cosine Similarity)

<br>

**Why?** To see if two words or sentences are "friends"—meaning they’re similar.
**What is it?** We use special math called *cosine similarity* to compare.
**Code:**

<br>

```python
from numpy.linalg import norm
A = np.array()[^1][^2][^3]
B = np.array()[^2][^3][^1]
cos_sim = A @ B / (norm(A) * norm(B))
print("Cosine Similarity (Math)", cos_sim)


from sklearn.metrics.pairwise import cosine_similarity
print("Cosine Similarity (Library)", cosine_similarity([A], [B]))
```

<br><br>





















<br><br>



<br>


## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br>


#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />


<br><br>

<p align="center">  ────────────── ⊹🔭๋ ──────────────

<!--
<p align="center">  ────────────── 🛸๋*ੈ✩* 🔭*ੈ₊ ──────────────
-->

<br>

<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>
  

  
#
 
##### <p align="center">Copyright 2025 Mindful-AI-Assistants. Code released under the  [MIT license.](https://github.com/Mindful-AI-Assistants/CDIA-Entrepreneurship-Soft-Skills-PUC-SP/blob/21961c2693169d461c6e05900e3d25e28a292297/LICENSE)






















