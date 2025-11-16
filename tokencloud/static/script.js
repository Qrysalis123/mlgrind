const colors = [
    '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
    '#FFB3E6', '#C9BAFF', '#FFBAD2', '#BAD7FF', '#BAFFBA',
    '#FFD4BA', '#E6BAFF', '#BAFFD4', '#FFDABA', '#D4BAFF',
    '#FFBAC9', '#BABFFF', '#FFFBBA', '#FFBABA', '#BAFFFF'
];

let debounceTimer;

function formatToken(token) {
    let formatted = token;
    formatted = formatted.replace(/\n/g, '\\n');
    formatted = formatted.replace(/\t/g, '\\t');
    formatted = formatted.replace(/ /g, '·');
    return formatted;
}

function createTokenRow(token, count) {
    const row = document.createElement('div');
    row.className = 'token-row';
    
    const tokenSpan = document.createElement('span');
    tokenSpan.className = 'token-text';
    tokenSpan.textContent = formatToken(token);
    
    const countSpan = document.createElement('span');
    countSpan.className = 'token-count';
    countSpan.textContent = count;
    
    row.appendChild(tokenSpan);
    row.appendChild(countSpan);
    return row;
}

async function generateCloud() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
        const text = document.getElementById('input').value;
        
        if (!text) {
            document.getElementById('tokens').innerHTML = '';
            document.getElementById('words').innerHTML = '';
            document.getElementById('special').innerHTML = '';
            document.getElementById('stats').textContent = '';
            return;
        }
        
        const response = await fetch('/tokenize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: text})
        });
        
        const data = await response.json();
        
        // Token sequence
        const tokensDiv = document.getElementById('tokens');
        tokensDiv.innerHTML = '';
        
        data.tokens.forEach((token, idx) => {
            const span = document.createElement('span');
            span.className = 'token';
            span.textContent = formatToken(token);
            span.style.backgroundColor = colors[idx % colors.length];
            span.title = 'Token ' + idx + ': "' + token + '" (ID: ' + data.token_ids[idx] + ')';
            tokensDiv.appendChild(span);
        });

        // Word tokens
        const wordsDiv = document.getElementById('words');
        wordsDiv.innerHTML = '';
        data.word_counts.forEach(([token, count]) => {
            wordsDiv.appendChild(createTokenRow(token, count));
        });

        // Special characters
        const specialDiv = document.getElementById('special');
        specialDiv.innerHTML = '';
        data.special_counts.forEach(([token, count]) => {
            specialDiv.appendChild(createTokenRow(token, count));
        });

        // Stats
        const uniqueWords = data.word_counts.length;
        const uniqueSpecial = data.special_counts.length;
        document.getElementById('stats').textContent = 
            'Total tokens: ' + data.tokens.length + 
            ' | Word tokens: ' + uniqueWords + 
            ' | Special chars: ' + uniqueSpecial + 
            ' | Characters: ' + text.length;
    }, 300);
}

document.getElementById('input').addEventListener('input', generateCloud);
generateCloud();
