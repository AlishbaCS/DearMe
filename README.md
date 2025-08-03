## DearMe – A Personalized Self-Reflection & Emotion-Aware Notebook

# 1) Overview
    
DearMe is a Jupyter Notebook that blends self-reflective journaling with machine learning. Users can engage with introspective writing prompts, while ML models help analyze emotions, tone, and sentiments to provide deeper self-awareness. This can support mental wellness, personal development, and emotional tracking over time.

# 2) File Structure
  ```
  DearMe/
  
  ├── DearMe.ipynb        # Main interactive notebook for journaling + ML analysis
  
  ├── README.md           # Project description and usage instructions
```

# 3) How to Use
  
  Install required packages:
  
    pip install notebook scikit-learn pandas matplotlib
  
  Launch the notebook:
  
    jupyter notebook DearMe.ipynb


Write your reflections and observe how ML models provide insights based on your writing.

# 4) Machine Learning Models Used
  
  a) Logistic Regression – Used for binary sentiment classification (positive vs. negative).
  
  
  b) Random Forest Classifier – Applied for multiclass emotion detection (e.g., joy, sadness, anger).
  
  
  c) Naive Bayes – Lightweight model for quick tone classification on small text inputs.
  

# 5) Handling Hyperparameters
  
  a) Hyperparameters were fine-tuned using:
  
  
  b) GridSearchCV – to search over predefined value combinations for each model.
  
  
  c) Cross-validation (CV=5) – to avoid overfitting and ensure generalization across journaling entries.
  

# 6) Key hyperparameters tuned:


  a) Logistic Regression: C, penalty
  
  
  b) Random Forest: n_estimators, max_depth
  
  
  c) Naive Bayes: smoothing parameter alpha
  


# 7) Gradient Descent Usage
   
Gradient descent was used in the training of Logistic Regression, where:


  a) The model minimizes the binary cross-entropy loss function:
  
  ```
  b) Loss = −1 𝑁∑𝑖=1 𝑁[𝑦 𝑖log⁡(𝑦^𝑖) + (1−𝑦𝑖)log⁡(1−𝑦^𝑖)] 
  ```
  
  c) Learning rate (alpha) and convergence criteria were adjusted for stability and performance.
  

# 8) Features

  a) Emotionally intelligent journaling prompts
  
  
  b) Sentiment/emotion analysis of your writing
  
  
  c) Gradient-based training for deeper emotional predictions
  
  
  d) 100% offline and private – your thoughts stay yours
  

# 9) Example Prompts

  What are you proud of today?
  
  
  What advice would you give your past self?
  
  
  What emotions are you carrying right now?
  

# 10) Use Cases

  i) Self-care routines
  
  
  ii) Emotional journaling with AI feedback
  
  
  iii) End-of-day reflection
  
  
  iv) Personal therapy supplement
  

# 11) Requirements

  Python 3.x and libraries
  
  Jupyter Notebook

  scikit-learn, pandas, matplotlib

# 12) 🙌 Acknowledgments

Crafted with care for anyone on a journey of healing, growth, or self-discovery—now powered by intelligent feedback through machine learning.

