# Topic Modeling on Arabic Dataset with BERTopic  

## 📌 Project Overview  
This project applies **BERTopic** for **topic modeling on Arabic news articles**.  
It leverages **sentence transformers for embedding generation, dimensionality reduction, clustering, and topic representation** to extract meaningful insights from unstructured text data.  

## 🚀 Features  
- **Arabic Text Preprocessing**: Cleaning, tokenization, and stop-word removal.  
- **Embedding Generation**: Uses **LaBSE** (Language-Agnostic BERT Sentence Embedding) for high-quality representations.  
- **Dimensionality Reduction**: Implements **UMAP** to reduce vector space while preserving topic structure.  
- **Clustering**: Utilizes **HDBSCAN** for robust unsupervised topic detection.  
- **Topic Representation**: Extracts **keywords using KeyBERT** for meaningful topic labels.  
- **Visualization**: Generates **topic heatmaps, hierarchical clustering, and evolution over time**.  
- **Inference Pipeline**: Allows topic assignment for new Arabic text.  
- **Model Persistence**: Saves and reloads trained models using **safetensors and pickle** formats.  

## 🛠️ Installation  
Ensure you have Python 3.8+ installed. Then, run:  
```bash
pip install bertopic==0.16.0 datasets==2.16.1 Arabic-Stopwords==0.4.3
pip install sentence-transformers umap-learn hdbscan

📂 Dataset
The project uses the Saudi News Net dataset from Hugging Face.
You can load it directly:

from datasets import load_dataset
dataset = load_dataset("saudinewsnet")
🏗️ How It Works
1️⃣ Preprocess Arabic text:
Removes URLs, numbers, punctuation.
Tokenizes and cleans text using NLTK.
2️⃣ Generate Embeddings:
Uses LaBSE from sentence-transformers.
3️⃣ Reduce Dimensions:
Applies UMAP for dimensionality reduction.
4️⃣ Cluster Topics:
Uses HDBSCAN to identify clusters.
5️⃣ Extract & Represent Topics:
Generates topic keywords using KeyBERT.
6️⃣ Visualize Topics:
Heatmaps, hierarchical structures, and topic evolution over time.
7️⃣ Inference:
Assigns new Arabic text to the most relevant topic.
🔍 Example Usage
Load & Preprocess Data

from datasets import load_dataset
import pandas as pd

dataset = load_dataset("saudinewsnet")
df = pd.DataFrame(dataset["train"])
df["text"] = df["content"].apply(clean_text)
Train BERTopic Model

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired

# Define models
embedding_model = SentenceTransformer("sentence-transformers/LaBSE")
umap_model = UMAP(n_components=15, metric="cosine", random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=50, prediction_data=True)
vectorizer = CountVectorizer(stop_words="arabic")
representation_model = {"KeyBERT": KeyBERTInspired()}

# Train BERTopic model
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    representation_model=representation_model,
    verbose=True
)

topics, probs = topic_model.fit_transform(df["text"].values)
Save & Load Model

topic_model.save("bertopic_arabic_model", serialization="safetensors")
loaded_model = BERTopic.load("bertopic_arabic_model")
Inference on New Text

story = "طرح مؤسسة البترول الكويتية عطاءً لبيع زيت الوقود عالي الكبريت"
topic, prob = topic_model.transform([story])
print("Predicted Topic:", topic)
📊 Visualization

topic_model.visualize_topics()
topic_model.visualize_heatmap()
topic_model.visualize_hierarchy()
topic_model.visualize_topics_over_time()
📌 Results
The model successfully identifies coherent topics from Arabic text.
Hierarchical clustering and topic merging allow deeper insights.
Inference pipeline enables assigning new Arabic documents to relevant topics.
📜 References
BERTopic Documentation
Sentence Transformers
Saudi News Net Dataset
📬 Contact
For questions or contributions, reach out at moab10107@gmail.com
