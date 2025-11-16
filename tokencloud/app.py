from flask import Flask, render_template, request, jsonify
import tiktoken
from collections import Counter
import string

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.json
    text = data.get('text', '')
    
    try:
        enc = tiktoken.get_encoding('gpt2')
        token_ids = enc.encode(text)
        tokens = [enc.decode([tid]) for tid in token_ids]
        
        # Separate into words and special characters
        words = []
        special_chars = []
        
        for token in tokens:
            stripped = token.strip()
            
            # Check if token is purely punctuation/whitespace/special chars OR single letter
            if not stripped or len(stripped) <= 1 or all(c in string.punctuation + string.whitespace for c in token):
                special_chars.append(token)
            else:
                words.append(token)
        
        # Count frequencies
        word_freq = Counter(words)
        special_freq = Counter(special_chars)
        
        # Sort by frequency (descending)
        word_counts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        special_counts = sorted(special_freq.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'tokens': tokens,
            'token_ids': token_ids,
            'word_counts': word_counts,
            'special_counts': special_counts
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Token Cloud Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
