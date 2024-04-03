from flask import Flask, request, render_template
import torch
from security_and_application_project import HybridAutoencoder, load_model, string_to_ascii_binary, binary_to_tensor, encrypted_vectors_to_string, string_to_encrypted_vectors, decrypt_encrypted_string

app = Flask(__name__)
model = load_model('hybrid_autoencoder_parameters.pth')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/encrypt', methods=['POST'])
def encrypt():
    text = request.form['text']
    ascii_binaries = string_to_ascii_binary(text)
    input_tensor = binary_to_tensor(ascii_binaries)

    encrypted_vectors = []
    with torch.no_grad():
        for sequence in input_tensor:
            encrypted_vector = model.encoder(sequence.unsqueeze(0))
            encrypted_vectors.append(encrypted_vector)
    encrypted_string = encrypted_vectors_to_string(encrypted_vectors)

    return render_template('encrypt.html', encrypted_text=encrypted_string)

@app.route('/decrypt', methods=['POST'])
def decrypt():
    encrypted_string = request.form['encrypted_text']
    decrypted_text = decrypt_encrypted_string(model, encrypted_string)
    return render_template('decrypt.html', decrypted_text=decrypted_text)

if __name__ == '__main__':
    app.run(debug=True)
