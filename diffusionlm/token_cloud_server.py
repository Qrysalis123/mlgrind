from flask import Flask, render_template_string, request, jsonify
import tiktoken

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Token Cloud</title>
    <style>
        body {
            font-family: monospace;
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
        }
        textarea {
            width: 100%;
            height: 150px;
            font-family: monospace;
            font-size: 14px;
            padding: 10px;
            border: 1px solid #ccc;
        }
        select, button {
            padding: 8px;
            font-size: 14px;
            margin: 10px 5px 10px 0;
        }
        #cloud {
            margin-top: 20px;
            line-height: 1.8;
        }
        .token {
            display: inline-block;
            padding: 4px 8px;
            margin: 3px;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }
        #stats {
            margin: 10px 0;
            color: #666;
        }
    </style>
</head>
<body>
    <h1>Token Cloud Visualizer</h1>
    
    <textarea id="input" placeholder="Type your text here...">Hello, I'm a language model and I can help you with many tasks!</textarea>
    
    <div>
        <select id="tokenizer">
            <option value="gpt2">gpt2</option>
            <option value="cl100k_base">cl100k_base (GPT-4)</option>
            <option value="p50k_base">p50k_base (Codex)</option>
            <option value="r50k_base">r50k_base (GPT-3)</option>
        </select>
        <button onclick="generateCloud()">Generate Token Cloud</button>
    </div>
    
    <div id="stats"></div>
    <div id="cloud"></div>

    <script>
        const colors = [
            '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
            '#FFB3E6', '#C9BAFF', '#FFBAD2', '#BAD7FF', '#BAFFBA',
            '#FFD4BA', '#E6BAFF', '#BAFFD4', '#FFDABA', '#D4BAFF',
            '#FFBAC9', '#BABFFF', '#FFFBBA', '#FFBABA', '#BAFFFF'
        ];

        async function generateCloud() {
            const text = document.getElementById('input').value;
            const tokenizer = document.getElementById('tokenizer').value;
            
            const response = await fetch('/tokenize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text, tokenizer: tokenizer})
            });
            
            const data = await response.json();
            
            const cloud = document.getElementById('cloud');
            cloud.innerHTML = '';
            
            data.tokens.forEach((token, idx) => {
                const span = document.createElement('span');
                span.className = 'token';
                span.textContent = token.replace(/\n/g, '\\\\n').replace(/\t/g, '\\\\t').replace(/ /g, '·');
                span.style.backgroundColor = colors[idx % colors.length];
                span.title = `Token ${idx}: "${token}" (ID: ${data.token_ids[idx]})`;
                cloud.appendChild(span);
            });

            document.getElementById('stats').textContent = 
                `Total tokens: ${data.tokens.length} | Characters: ${text.length}`;
        }

        generateCloud();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.json
    text = data.get('text', '')
    tokenizer_name = data.get('tokenizer', 'gpt2')
    
    try:
        enc = tiktoken.get_encoding(tokenizer_name)
        token_ids = enc.encode(text)
        tokens = [enc.decode([tid]) for tid in token_ids]
        
        return jsonify({
            'tokens': tokens,
            'token_ids': token_ids
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Token Cloud Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
