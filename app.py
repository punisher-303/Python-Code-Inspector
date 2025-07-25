import gradio as gr
import joblib
import numpy as np
import re
from tokenizer import tokenize_code
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
import logging
from datetime import datetime
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1. Enhanced Vectorizer Loading with Validation
def load_vectorizer():
    """Safely load and reconstruct the vectorizer with validation checks"""
    try:
        vec_data = joblib.load("vectorizer.joblib")
        
        # Validate required keys
        required_keys = {'vocabulary_', 'idf_', 'ngram_range', 'analyzer'}
        if not all(key in vec_data for key in required_keys):
            raise ValueError("Missing required vectorizer attributes")
        
        # Create new vectorizer
        vectorizer = TfidfVectorizer(
            max_features=vec_data.get('max_features', 200),
            ngram_range=vec_data['ngram_range'],
            analyzer=vec_data['analyzer'],
            stop_words=vec_data.get('stop_words', None),
            min_df=vec_data.get('min_df', 1)
        )
        
        # Restore fitted attributes
        vectorizer.vocabulary_ = vec_data['vocabulary_']
        vectorizer.idf_ = vec_data['idf_']
        vectorizer.fixed_vocabulary_ = True
        
        # Additional validation
        if len(vectorizer.vocabulary_) == 0:
            raise ValueError("Empty vocabulary loaded")
            
        return vectorizer
        
    except Exception as e:
        logger.error(f"Vectorizer loading failed: {str(e)}")
        raise gr.Error(f"Vectorizer loading failed. Please retrain your model.\nError: {str(e)}")

# 2. Enhanced Model Loading
try:
    model_data = joblib.load("model.joblib")
    model = model_data['model']
    vectorizer = load_vectorizer()
    model_metadata = model_data.get('metadata', {})
    
    logger.info(f"Model loaded successfully. Training date: {model_metadata.get('training_date', 'unknown')}")
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
except Exception as e:
    logger.critical(f"Model loading failed: {str(e)}")
    raise gr.Error(f"""Model loading failed. Please check:
    1. Files exist in: {os.listdir('.')}
    2. Matching package versions
    3. File integrity
    Error: {str(e)}""")

# 3. Enhanced Correction System with More Patterns
CORRECTION_RULES = [
    (r'def (\w+)\(([^)]*)\):\s*return\s+([^=]+)\s*-\s*([^\s;]+)', r'def \1(\2):\n    return \3 + \4'),
    (r'if\s+(\w+)\s*=\s*([^:]+):', r'if \1 == \2:'),
    (r'for\s+(\w+)\s+in\s+range\(([^)]+)\)\s+(\w+)', r'for \1 in range(\2): \3'),
    (r'print\s+[\'"](.+?)[\'"]', r'print("\1")'),
    (r'while\s+(True|\w+)\s+(\w+)', r'while \1: \2'),
    (r'\[([^\]]+)\s+([^\]]+)\s+([^\]]+)\]', r'[\1, \2, \3]'),
    (r'\{\s*([^\s}]+)\s+([^\s}]+)\s*\}', r'{"\1": "\2"}'),
    (r'class\s+(\w+)([^\n:]+)', r'class \1:'),
    (r'except\s+(\w+)([^\n:]+)', r'except \1:'),
    (r'=\s*[\'"]', r'= "'),  # Standardize string assignments
    (r'([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*)\s*([+-/*])\s*([a-zA-Z_]\w*)', r'\1 = \2 \3 \4'),  # Space around operators
    (r'import\s+([a-zA-Z_]\w*)\s*,\s*([a-zA-Z_]\w*)', r'import \1\nimport \2'),  # Split imports
    (r'([^#\n])\s*#\s*([^\n]*)', r'\1  # \2')  # Standardize comments
]

def suggest_correction(code):
    """Apply pattern-based corrections with line context and confidence"""
    lines = code.split('\n')
    corrected_lines = []
    correction_log = []
    
    for i, line in enumerate(lines):
        original_line = line.strip()
        corrected_line = line
        
        for pattern, replacement in CORRECTION_RULES:
            if re.search(pattern, line):
                corrected_line = re.sub(pattern, replacement, line)
                if corrected_line != line:
                    correction_log.append({
                        'line': i+1,
                        'original': original_line,
                        'corrected': corrected_line.strip(),
                        'rule': pattern
                    })
                break
                
        corrected_lines.append(corrected_line)
    
    return '\n'.join(corrected_lines), correction_log

# 4. Fixed Prediction Function
def predict(code):
    """Analyze Python code with detailed feedback"""
    if not code.strip():
        return (
            "🟠 No code entered",  # analysis_output
            "",                    # corrected_output
            0,                     # confidence_meter
            {"error": "No code entered"}  # details_output
        )
    
    try:
        # Tokenization
        tokens = tokenize_code(code, normalize=True)
        if not tokens.strip() or "TOKENIZE_ERROR" in tokens:
            return (
                "🔴 Invalid Python syntax",
                code,
                0,
                {"error": "Tokenization failed"}
            )
        
        # Vectorization
        X = vectorizer.transform([tokens])
        
        # Prediction
        pred = model.predict(X)[0]
        confidence = float(model.predict_proba(X)[0][pred]) * 100  # Convert to percentage
        
        # Generate correction if needed
        if pred == 1:  # Bug detected
            corrected_code, corrections = suggest_correction(code)
            return (
                f"⚠️ Potential Bug ({confidence:.0f}% confidence)",
                corrected_code,
                confidence,
                {
                    "prediction": "potential_bug",
                    "confidence": confidence/100,
                    "tokenized": tokens,
                    "corrections": corrections,
                    "system_stats": {
                        "memory": psutil.virtual_memory().percent,
                        "load": os.getloadavg()[0]
                    }
                }
            )
        else:
            return (
                f"✅ Code Looks Good ({confidence:.0f}% confidence)",
                code,
                confidence,
                {
                    "prediction": "clean_code",
                    "confidence": confidence/100,
                    "tokenized": tokens,
                    "system_stats": {
                        "memory": psutil.virtual_memory().percent,
                        "load": os.getloadavg()[0]
                    }
                }
            )
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return (
            f"🔴 Error: {str(e)}",
            code,
            0,
            {"error": str(e)}
        )

# 5. Fixed Gradio Interface
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Python Bug Detector") as demo:
        gr.Markdown("""# 🐞 Python Code Inspector""")
        
        # System Info Section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### System Status")
                sys_info = gr.JSON(
                    label="System Information",
                    value={
                        "python_version": sys.version,
                        "system": os.uname().sysname,
                        "model_date": model_metadata.get("training_date", "unknown"),
                        "vocab_size": len(vectorizer.vocabulary_)
                    }
                )
                update_btn = gr.Button("Refresh System Info")
                
            with gr.Column(scale=3):
                gr.Markdown("### Code Analysis")
        
        # Main Analysis Interface
        with gr.Row():
            with gr.Column():
                input_code = gr.Code(
                    label="Your Python Code",
                    language="python",
                    lines=10,
                    interactive=True,
                    value="def add(a, b): return a - b"  # Default example
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Column():
                analysis_output = gr.Textbox(
                    label="Analysis Result",
                    interactive=False
                )
                confidence_meter = gr.Slider(
                    label="Confidence Level",
                    minimum=0,
                    maximum=100,
                    interactive=False
                )
                corrected_output = gr.Code(
                    label="Suggested Correction", 
                    language="python", 
                    lines=10,
                    interactive=True
                )
        
        # Examples Section
        gr.Examples(
            examples=[
                ["def add(a, b): return a - b"],
                ["if x = 5: pass"],
                ["for i in range(10) print(i)"],
                ["print 'Hello'"],
                ["while True print('Loop')"],
                ["lst = [1 2 3]"],
                ["dict = {'key' 'value'}"]
            ],
            inputs=input_code,
            label="Common Bug Examples"
        )
        
        # Event Handlers
        def update_sys_info():
            return {
                "python_version": sys.version,
                "system": os.uname().sysname,
                "memory": f"{psutil.virtual_memory().percent}% used",
                "load": os.getloadavg()[0],
                "model_date": model_metadata.get("training_date", "unknown"),
                "vocab_size": len(vectorizer.vocabulary_),
                "timestamp": datetime.now().isoformat()
            }
        
        update_btn.click(
            fn=update_sys_info,
            outputs=sys_info
        )
        
        analyze_btn.click(
            fn=predict,
            inputs=input_code,
            outputs=[analysis_output, corrected_output, confidence_meter]
        )
        
        return demo

# Run the application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
        debug=False
    )