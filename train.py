import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from tokenizer import tokenize_code
import hashlib
import psutil
import sklearn  # Add this with other imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1. Enhanced Training Data with More Examples
def load_training_data():
    """Load and balance training data with validation"""
    buggy_examples = [
        "def add(a, b): return a - b",  # Wrong operator
        "if x = 5: pass",  # Assignment instead of comparison
        "for i in range(10) print(i)",  # Missing colon
        "print 'Hello'",  # Python 2 print statement
        "def divide(a, b): return a * b",  # Wrong operator
        "while True print('Loop')",  # Missing colon
        "lst = [1 2 3]",  # Missing commas
        "dict = {'key' 'value'}",  # Missing colon
        "class MyClass\n    pass",  # Missing colon
        "try:\n    x=1\nexcept ValueError\n    pass",  # Missing colon
        "def func() return 42",  # Missing colon
        "with open(file) as f print(f.read())",  # Missing colon
        "[x for x in range(10) if x%2==0]",  # Unbalanced paren
        "a = b + c",  # Undefined variables
        "import os; os.system('rm -rf /')",  # Dangerous operation
        "return 42",  # Return outside function
        "break",  # Break outside loop
        "x == None",  # Should use 'is' for None
        "if len(mylist) > 0:",  # Redundant condition
        "except:",  # Bare except
    ]

    correct_examples = [
        "def add(a, b): return a + b",
        "if x == 5: pass",
        "for i in range(10): print(i)",
        "print('Hello')",
        "def divide(a, b): return a / b",
        "while True: print('Loop')",
        "lst = [1, 2, 3]",
        "dict = {'key': 'value'}",
        "class MyClass:\n    pass",
        "try:\n    x=1\nexcept ValueError:\n    pass",
        "def func(): return 42",
        "with open(file) as f: print(f.read())",
        "[x for x in range(10) if x%2==0]",
        "a = b + c  # Assuming b and c are defined",
        "# Removed dangerous operation",
        "def foo():\n    return 42",
        "while True:\n    break",
        "x is None",
        "if mylist:",
        "except Exception as e:"
    ]

    # Validate data balance
    assert len(buggy_examples) == len(correct_examples), "Dataset imbalance detected"
    
    # Tokenize all examples
    X = [tokenize_code(code) for code in buggy_examples + correct_examples]
    y = [1] * len(buggy_examples) + [0] * len(correct_examples)
    
    return X, y

# 2. Enhanced Vectorizer Configuration
def create_vectorizer():
    """Create and configure the TF-IDF vectorizer"""
    return TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 4),
        stop_words=None,
        analyzer='word',
        token_pattern=r'\b\w+\b|\S',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        smooth_idf=True
    )

# 3. Enhanced Model Configuration
def create_model():
    """Create and configure the Random Forest model"""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbose=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True
    )

# 4. Enhanced Model Evaluation
def evaluate_model(model, X, y):
    """Perform comprehensive model evaluation"""
    logger.info("\n=== Model Evaluation ===")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV accuracy: {np.mean(cv_scores):.1%} ± {np.std(cv_scores):.1%}")
    
    # Train-test split evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Classification report
    y_pred = model.predict(X_test)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['Correct', 'Buggy']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(cm)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_names = vectorizer.get_feature_names_out()
        top_features = sorted(zip(feature_names, model.feature_importances_), 
                           key=lambda x: x[1], reverse=True)[:20]
        logger.info("\nTop 20 Important Features:")
        for feat, imp in top_features:
            logger.info(f"{feat}: {imp:.4f}")

# 5. Enhanced Model Saving with Metadata
def save_models(model, vectorizer, X_train):
    """Save models with comprehensive metadata"""
    # Prepare vectorizer data
    vectorizer_data = {
        'vocabulary_': vectorizer.vocabulary_,
        'idf_': vectorizer.idf_,
        'stop_words_': vectorizer.stop_words,
        'fixed_vocabulary_': vectorizer.fixed_vocabulary_,
        'ngram_range': vectorizer.ngram_range,
        'analyzer': vectorizer.analyzer,
        'max_features': vectorizer.max_features,
        'min_df': vectorizer.min_df,
        'max_df': vectorizer.max_df,
        '_tfidf': vectorizer._tfidf
    }
    
    # Prepare model metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'num_samples': len(X_train),
        'classes_distribution': np.bincount(y_train),
        'system': {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': os.cpu_count(),
            'memory': psutil.virtual_memory().percent
        },
        'dependencies': {
            'scikit-learn': sklearn.__version__,
            'numpy': np.__version__,
            'joblib': joblib.__version__
        },
        'parameters': {
            'model': model.get_params(),
            'vectorizer': vectorizer.get_params()
        }
    }
    
    # Save with compression
    joblib.dump(vectorizer_data, "vectorizer.joblib", compress=3)
    joblib.dump({'model': model, 'metadata': metadata}, "model.joblib", compress=3)
    
    # Generate checksums
    def file_checksum(filename):
        with open(filename, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    logger.info("\n=== Model Files ===")
    for fname in ["vectorizer.joblib", "model.joblib"]:
        size = os.path.getsize(fname) / 1024
        checksum = file_checksum(fname)
        logger.info(f"{fname}: {size:.1f} KB | SHA256: {checksum[:16]}...")

# Main Training Pipeline
if __name__ == "__main__":
    try:
        logger.info("=== Starting Training Process ===")
        
        # 1. Load and prepare data
        X_train, y_train = load_training_data()
        logger.info(f"Loaded {len(X_train)} training examples")
        
        # 2. Create and fit vectorizer
        vectorizer = create_vectorizer()
        X_vectorized = vectorizer.fit_transform(X_train)
        logger.info(f"Vectorizer created with {len(vectorizer.vocabulary_)} features")
        
        # 3. Create and train model
        model = create_model()
        model.fit(X_vectorized, y_train)
        logger.info("Model training completed")
        
        # 4. Evaluate model
        evaluate_model(model, X_vectorized, y_train)
        
        # 5. Save models
        save_models(model, vectorizer, X_train)
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error("Training failed!", exc_info=True)
        sys.exit(1)