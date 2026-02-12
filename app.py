import gradio as gr
import torch
import torch.nn as nn
import pickle
import time
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet once globally (PERFORMANCE FIX)
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)
resnet.eval()
class Encoder(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.fc = nn.Linear(2048, hidden_dim)
        self.relu = nn.ReLU()  # ARCHITECTURE FIX: Added ReLU
    
    def forward(self, features):
        return self.relu(self.fc(features))  # ARCHITECTURE FIX: Apply ReLU

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, word, encoder_hidden, hidden=None):
        embed = self.dropout(self.embedding(word))
        if hidden is None:
            h0 = encoder_hidden.unsqueeze(0)
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)
        output, hidden = self.lstm(embed, hidden)
        output = self.fc(output)
        return output, hidden

class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(hidden_dim)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, dropout)

    # CRITICAL FIX: Added forward() method to match training architecture
    def forward(self, image_features, captions, hidden=None):
        enc_out = self.encoder(image_features)
        outputs, hidden = self.decoder(captions, enc_out, hidden)
        return outputs, hidden

def load_vocab():
    with open('vocab.pkl', 'rb') as f:
        vocab_dict = pickle.load(f)
    return vocab_dict['word2idx'], vocab_dict['idx2word']

def load_model(vocab_size):
    model = CaptionModel(vocab_size).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    return model

def extract_features(image):
    # PERFORMANCE FIX: ResNet now loaded globally, not recreated each time
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.fromarray(image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = resnet(img_tensor)
    return features.view(features.size(0), -1)

def greedy_caption(model, features, word2idx, idx2word, max_len=20):
    model.eval()
    with torch.no_grad():
        encoder_hidden = model.encoder(features)
        word = torch.tensor([[word2idx['<start>']]], device=device)
        hidden = None
        caption = []
        
        for _ in range(max_len):
            output, hidden = model.decoder(word, encoder_hidden, hidden)
            # SHAPE FIX: Correct argmax operation
            predicted = output[:, -1, :].argmax(-1)
            word_idx = predicted.item()
            
            if word_idx == word2idx['<end>']:
                break
            
            word_str = idx2word[word_idx]
            if word_str not in ['<start>', '<pad>', '<unk>']:
                caption.append(word_str)
            
            word = torch.tensor([[word_idx]], device=device)
    
    return ' '.join(caption)

def beam_search_caption(model, features, word2idx, idx2word, beam_width=5, max_len=20):
    model.eval()
    end_idx = word2idx['<end>']
    start_idx = word2idx['<start>']
    
    with torch.no_grad():
        encoder_hidden = model.encoder(features)
    
    beams = [([start_idx], 0.0, None)]
    completed = []
    
    for _ in range(max_len):
        if not beams:
            break
        
        candidates = []
        for seq, score, hidden in beams:
            if seq[-1] == end_idx:
                completed.append((seq, score))
                continue
            
            word = torch.tensor([[seq[-1]]], device=device)
            with torch.no_grad():
                output, new_hidden = model.decoder(word, encoder_hidden, hidden)
            
            log_probs = torch.log_softmax(output[:, -1, :], dim=-1)
            topk_probs, topk_idxs = log_probs.topk(beam_width)
            
            for i in range(beam_width):
                next_word = topk_idxs[0, i].item()
                next_score = score + topk_probs[0, i].item()
                
                if next_word == seq[-1]:
                    next_score -= 1e9
                
                new_seq = seq + [next_word]
                # CRITICAL FIX: Clone hidden states for each beam
                candidates.append((new_seq, next_score, (
                    new_hidden[0].clone(),
                    new_hidden[1].clone()
                )))
        
        candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        beams = candidates[:beam_width]
    
    if completed:
        best_seq = max(completed, key=lambda x: x[1] / len(x[0]))[0]
    else:
        best_seq = beams[0][0] if beams else [start_idx]
    
    caption = []
    for idx in best_seq[1:]:
        if idx == end_idx:
            break
        word = idx2word.get(idx, '')
        if word and word not in ['<start>', '<pad>', '<unk>']:
            caption.append(word)
    
    return ' '.join(caption)

def generate_captions(image, progress=gr.Progress()):
    """Generate captions without animation"""
    if image is None:
        return None, "Please upload an image first...", "Please upload an image first..."
    
    progress(0, desc="Extracting image features...")
    features = extract_features(image)
    
    progress(0.3, desc="Generating greedy caption...")
    greedy = greedy_caption(model, features, word2idx, idx2word)
    
    progress(0.6, desc="Generating beam search caption...")
    beam = beam_search_caption(model, features, word2idx, idx2word)
    
    progress(0.9, desc="Formatting captions...")
    
    # Capitalize first letter and add period if not present
    greedy = greedy[0].upper() + greedy[1:] if greedy else ""
    beam = beam[0].upper() + beam[1:] if beam else ""
    
    if greedy and not greedy.endswith('.'):
        greedy += '.'
    if beam and not beam.endswith('.'):
        beam += '.'
    
    progress(1.0, desc="Done!")
    
    return image, greedy, beam

word2idx, idx2word = load_vocab()
model = load_model(len(word2idx))

css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

.gradio-container {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

#title {
    text-align: center;
    background: linear-gradient(90deg, #00ff00, #00cc88, #00ff00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3em;
    font-weight: 900;
    margin: 20px 0 30px 0;
    font-family: 'Orbitron', monospace;
    text-transform: uppercase;
    letter-spacing: 3px;
    text-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    from {
        filter: drop-shadow(0 0 5px #00ff00);
    }
    to {
        filter: drop-shadow(0 0 20px #00ff00);
    }
}

@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}

.output-caption {
    background: linear-gradient(145deg, #1a1a1a, #0f0f0f) !important;
    border: 2px solid #00ff00 !important;
    border-radius: 8px !important;
    color: #00ff88 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.15em !important;
    padding: 20px !important;
    box-shadow: 0 0 15px rgba(0, 255, 0, 0.2), inset 0 0 10px rgba(0, 255, 0, 0.05) !important;
    line-height: 1.6 !important;
    min-height: 60px !important;
    transition: all 0.3s ease !important;
}

.output-caption:hover {
    border-color: #00ff88 !important;
    box-shadow: 0 0 25px rgba(0, 255, 0, 0.4), inset 0 0 15px rgba(0, 255, 0, 0.1) !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #00ff00, #00cc88) !important;
    color: #0a0a0a !important;
    border: none !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: bold !important;
    font-size: 1.1em !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 255, 0, 0.3) !important;
}

.gr-button-primary:hover {
    background: linear-gradient(135deg, #00ff88, #00ff00) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 255, 0, 0.5) !important;
}

.gr-button-primary:active {
    transform: translateY(0px) !important;
}

label {
    color: #00ff88 !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    font-size: 0.95em !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 8px !important;
}

.image-container {
    border: 2px solid #00ff00 !important;
    border-radius: 8px !important;
    padding: 10px !important;
    background: linear-gradient(145deg, #1a1a1a, #0f0f0f) !important;
    box-shadow: 0 0 15px rgba(0, 255, 0, 0.2) !important;
}

/* Make images fit properly in both frames */
#image_input, #image_output {
    height: 380px !important;
}

#image_input img, #image_output img {
    max-height: 360px !important;
    max-width: 100% !important;
    height: auto !important;
    width: auto !important;
    object-fit: contain !important;
}

#image_input > div, #image_output > div {
    height: 380px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Ensure image wrapper doesn't overflow */
#image_input .image-frame, #image_output .image-frame {
    height: 360px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

#image_input .image-container, #image_output .image-container {
    height: 360px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Styling for the image upload area */
.gr-file-input {
    border: 2px dashed #00ff00 !important;
    border-radius: 8px !important;
    background: rgba(0, 255, 0, 0.05) !important;
}

/* Add subtle scan line effect */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: repeating-linear-gradient(
        0deg,
        rgba(0, 255, 0, 0.03) 0px,
        transparent 1px,
        transparent 2px,
        rgba(0, 255, 0, 0.03) 3px
    );
    pointer-events: none;
    z-index: 1;
}

/* Subtitle styling */
.subtitle {
    text-align: center;
    color: #00cc88;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1em;
    margin-bottom: 30px;
    opacity: 0.8;
}
"""

with gr.Blocks() as app:
    gr.HTML('<div id="title">Neural Storyteller</div>')
    gr.HTML('<div class="subtitle">AI-Powered Image Caption Generator • Seq2Seq Architecture</div>')
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Image", 
                type="numpy",
                height=380,
                elem_id="image_input"
            )
            generate_btn = gr.Button("Generate Captions", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            image_output = gr.Image(
                label="Preview", 
                height=380,
                elem_id="image_output"
            )
            
    with gr.Row():
        with gr.Column():
            greedy_output = gr.Textbox(
                label="Greedy Decoding",
                elem_classes="output-caption",
                lines=3,
                interactive=False,
                placeholder="Caption will appear here..."
            )
        
        with gr.Column():
            beam_output = gr.Textbox(
                label="Beam Search Decoding",
                elem_classes="output-caption", 
                lines=3,
                interactive=False,
                placeholder="Caption will appear here..."
            )
    
    generate_btn.click(
        fn=generate_captions,
        inputs=image_input,
        outputs=[image_output, greedy_output, beam_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, css=css, theme=gr.themes.Monochrome())