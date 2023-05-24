import openai
from transformers import GPT2Tokenizer, GPT2Model

# Step 1: Pengumpulan Data
# Collect training data
training_data = [
    "User: Halo",
    "Chatbot: Hai, ada yang bisa saya bantu?",
    "User: Bagaimana cara memesan produk?",
    "Chatbot: Anda dapat memesan produk kami melalui situs web kami. Silakan kunjungi halaman pemesanan dan ikuti langkah-langkahnya.",
    # tambahkan lebih banyak dialog training di sini
]

# Step 2: Pelatihan Model GPT
# Inisialisasi tokenizer dan model GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Tokenisasi data pelatihan
tokenized_training_data = tokenizer(training_data, truncation=True, padding=True)

# Melatih model
model.train(tokenized_training_data)

# Step 3: Simpan Model
save_directory = "path/to/save_directory"

# Simpan model dan tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
