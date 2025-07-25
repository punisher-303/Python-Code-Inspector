import tokenize
from io import BytesIO
import re
import logging
from keyword import iskeyword
from tokenize import tok_name
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
IGNORE_TOKENS = {
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NL
}

SPECIAL_HANDLING = {
    tokenize.STRING: "STRING_LITERAL",
    tokenize.NUMBER: "NUMERIC_LITERAL",
    tokenize.COMMENT: "COMMENT",
    tokenize.NAME: "IDENTIFIER"
}

PYTHON_KEYWORDS = {
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 
    'await', 'break', 'class', 'continue', 'def', 'del', 
    'elif', 'else', 'except', 'finally', 'for', 'from', 
    'global', 'if', 'import', 'in', 'is', 'lambda', 
    'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 
    'try', 'while', 'with', 'yield'
}

def tokenize_code(code, include_types=False, keep_comments=False, normalize=True):
    """
    Enhanced Python code tokenizer with multiple processing modes and robust error handling.
    
    Args:
        code (str): Python source code to tokenize
        include_types (bool): Whether to include token type information
        keep_comments (bool): Whether to preserve comments
        normalize (bool): Whether to normalize literals and identifiers
        
    Returns:
        str: Space-separated string of tokens with optional metadata
    """
    try:
        # First attempt with Python's built-in tokenizer
        tokens = []
        code_bytes = code.encode('utf-8')
        
        for tok in tokenize.tokenize(BytesIO(code_bytes).readline):
            token_type = tok.type
            token_string = tok.string
            
            # Skip ignored tokens
            if token_type in IGNORE_TOKENS:
                continue
                
            # Handle comments
            if token_type == tokenize.COMMENT:
                if keep_comments:
                    tokens.append(f"COMMENT:{token_string}" if include_types else token_string)
                continue
                
            # Apply special handling
            if normalize and token_type in SPECIAL_HANDLING:
                if token_type == tokenize.NAME and iskeyword(token_string):
                    token_string = f"KEYWORD_{token_string.upper()}"
                elif token_type == tokenize.NAME and token_string in PYTHON_KEYWORDS:
                    token_string = f"PYKEYWORD_{token_string}"
                elif token_type in SPECIAL_HANDLING:
                    token_string = SPECIAL_HANDLING[token_type]
            
            # Format output based on requirements
            if include_types:
                token_str = f"{token_string}:{tok_name[token_type]}"
            else:
                token_str = token_string
                
            tokens.append(token_str)
            
        return " ".join(tokens)
    
    except tokenize.TokenError as e:
        # Handle partial/incomplete code
        logger.warning(f"Token error: {e}, using fallback tokenization")
        return fallback_tokenize(code)
        
    except IndentationError as e:
        logger.warning(f"Indentation error: {e}")
        return f"INDENT_ERROR {fallback_tokenize(code)}"
        
    except Exception as e:
        logger.error(f"Unexpected tokenization error: {e}")
        return f"TOKENIZE_ERROR {fallback_tokenize(code)}"

def fallback_tokenize(code):
    """
    Robust fallback tokenization when standard tokenizer fails.
    Handles partial code, syntax errors, and other edge cases.
    """
    # Pre-process special patterns
    code = re.sub(r'([\'"]).*?\1', ' STRING_LITERAL ', code)  # String literals
    code = re.sub(r'\b\d+\.?\d*\b', ' NUMERIC_LITERAL ', code)  # Numbers
    code = re.sub(r'#[^\n]*', ' COMMENT ', code)  # Comments
    
    # Handle common Python syntax elements
    code = re.sub(r'([\[\](){},.:;@])', r' \1 ', code)  # Punctuation
    code = re.sub(r'([=!<>]=|[+\-*/%]=|->|\*\*)', r' \1 ', code)  # Compound operators
    code = re.sub(r'([=!<>+\-*/%&|^~])', r' \1 ', code)  # Single operators
    
    # Normalize whitespace and clean up
    tokens = code.split()
    normalized = []
    
    for token in tokens:
        if token in PYTHON_KEYWORDS:
            normalized.append(f"PYKEYWORD_{token}")
        elif token.isidentifier():
            normalized.append("IDENTIFIER")
        else:
            normalized.append(token)
    
    return " ".join(normalized)

def tokenize_file(file_path, **kwargs):
    """
    Tokenize an entire Python source file with line number tracking
    
    Args:
        file_path (str): Path to Python file
        **kwargs: Arguments to pass to tokenize_code
        
    Returns:
        dict: {line_number: tokenized_line}
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        return {
            i+1: tokenize_code(line, **kwargs)
            for i, line in enumerate(lines)
            if line.strip()
        }
    except Exception as e:
        logger.error(f"Failed to tokenize file {file_path}: {e}")
        return {}

# Test cases for validation
TEST_CASES = [
    ("def add(a, b): return a + b", "def IDENTIFIER ( IDENTIFIER , IDENTIFIER ) : return IDENTIFIER + IDENTIFIER"),
    ("if x = 5: pass", "if IDENTIFIER = NUMERIC_LITERAL : PYKEYWORD_pass"),
    ("print('hello')", "IDENTIFIER ( STRING_LITERAL )"),
    ("# Comment\nx = 1", "IDENTIFIER = NUMERIC_LITERAL"),
    ("lst = [1, 2, 3]", "IDENTIFIER = [ NUMERIC_LITERAL , NUMERIC_LITERAL , NUMERIC_LITERAL ]")
]

def run_tests():
    """Validate tokenizer against test cases"""
    failures = 0
    for code, expected in TEST_CASES:
        result = tokenize_code(code, normalize=True)
        if result != expected:
            logger.error(f"Test failed:\nInput: {code}\nExpected: {expected}\nGot: {result}")
            failures += 1
    
    if failures == 0:
        logger.info("✅ All tokenizer tests passed!")
    else:
        logger.warning(f"⚠️ {failures}/{len(TEST_CASES)} tests failed")

if __name__ == "__main__":
    logger.info("Running tokenizer tests...")
    run_tests()