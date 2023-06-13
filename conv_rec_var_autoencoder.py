import jieba
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchtext as tt
import torchvision as tv

from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator, GloVe

# ========== Hyperparameters ==========
batch_size = 16
device = "cuda"
hidden_layer_dim = 512
img_height, img_width = (120, 200)
lambd = 3
latent_layer_dim = 1000
lr = 1e-4
num_epochs = 500
word_embed_dim = 300

torch.manual_seed(3407)

# ========== Paths ==========
source = "us"
image_dir = "input_data/" + source + "/"
data_save_dir = "output_data/" + source + "/"

# ========== Tokens ==========
pad_token = "<PAD>"
pad_value = 0
sos_token = "<SOS>"
sos_value = 1

# ========== Data Preprocessing ==========
if source == "cn":
    data = pd.read_csv("input_data/cn.txt", header = None)
    data.rename({0: "text"}, axis = 1, inplace = True)
else:
    data = pd.read_csv("input_data/us.txt", index_col = 0)

data["image_path"] = data.index.astype(str)
data["image_path"] = data.apply(lambda x: image_dir + x.image_path + ".jpg", axis = 1)
print(data)

chn_punct = {"，", "。", "？"}
if source == "cn":
    tokenizer = lambda text: [word for word in jieba.lcut(text) if word not in chn_punct]
    print(tokenizer("你好世界"))
else:
    tokenizer = tt.data.utils.get_tokenizer("basic_english")
    print(tokenizer("hello world"))


def yield_tokens(data_iter):
    """
    Tokenize each sentence to construct vocabulary set
    """
    for text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(data.text), specials = [pad_token, sos_token])
resize = tv.transforms.Resize((img_height, img_width))


def preprocess_text(batch):
    """
    Transform raw text into encoded and padded tensors
    """
    text_list = []
    for text in batch:
        text_encoded = torch.tensor(vocab.lookup_indices(tokenizer(text)), dtype = torch.int64)
        text_list.append(text_encoded.clone().detach())

    text_tensor = torch.nn.utils.rnn.pad_sequence(text_list, batch_first = True, padding_value = pad_value)
    return text_tensor.to(device)


def preprocess_image(batch):
    """
    Transform raw image into resized and normalized tensors
    """
    image_list = []
    for path in batch:
        img = tv.io.read_image(path)
        img = resize(img)
        img = img / 255.0
        image_list.append(img)

    image_tensor = torch.stack(image_list)
    return image_tensor.to(device)


data_text_batch = DataLoader(data.text, batch_size = batch_size, shuffle = False, collate_fn = preprocess_text)
data_image_batch = DataLoader(data.image_path, batch_size = batch_size, shuffle = False, collate_fn = preprocess_image)

# Check the correctness of dataset
for text, image in zip(data_text_batch, data_image_batch):
    print(text.shape)
    print(image.shape)
    print(vocab.lookup_tokens(list(text[0])))
    plt.imshow(np.transpose(image[0].cpu().numpy(), (1, 2, 0)))
    plt.axis("off")
    break


glove = GloVe(dim = word_embed_dim)
print(glove["coronavirus"])

chn_word2vec = {}
with open("input_data/sgns.wiki.bigram-char", "r", encoding = "utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
            chn_word2vec[word] = torch.tensor(np.asarray(values[1:], dtype = np.float32))
        except ValueError as ve:
            print("Error in embedding word:", word)
            print(ve)

print(chn_word2vec["疫苗"])


# ========== Model Construction ==========
class Encoder(nn.Module):
    def __init__(self, embed_layer, lstm_n_layers):
        super(Encoder, self).__init__()
        # Text
        self.n_layers = lstm_n_layers
        self.embed = embed_layer
        self.lstm = nn.LSTM(word_embed_dim, hidden_layer_dim, num_layers = self.n_layers, batch_first = True, bidirectional = True)
        self.bn1 = nn.BatchNorm1d(4 * self.n_layers * hidden_layer_dim)

        # Image
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = "same")
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, padding = "same")
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3, padding = "same")
        self.maxpool = nn.MaxPool2d((2, 2))
        self.bn2 = nn.BatchNorm1d(img_height * img_width // 2)

        self.d1 = nn.Linear(4 * self.n_layers * hidden_layer_dim + img_height * img_width // 2, 4000)
        self.out = nn.Linear(4000, 2 * latent_layer_dim)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, input_text, input_image, hidden_cell):
        # Text
        embed_text = self.embed(input_text)  # (batch_size, seq_len, word_embed_dim)
        _, (hidden, cell) = self.lstm(embed_text, hidden_cell)  # (4, batch_size, hidden_layer_dim)
        x_text = torch.transpose(torch.cat([hidden, cell], dim = 0), 0, 1)  # (batch_size, 8, hidden_layer_dim)
        x_text = self.bn1(self.flat(x_text))  # (batch_size, 8 * hidden_layer_dim)

        # Image
        img = self.relu(self.conv1(input_image))  # (batch_size, 32, img_height, img_width)
        img = self.maxpool(img)  # (batch_size, 32, img_height / 2, img_width / 2)
        img = self.relu(self.conv2(img))
        img = self.maxpool(img)
        img = self.relu(self.conv3(img))
        img = self.maxpool(img)  # (batch_size, 32, img_height / 8, img_width / 8)
        x_image = self.bn2(self.flat(img))  # (batch_size, img_height * img_width / 2)

        x = torch.cat([x_text, x_image], dim = 1)
        x = self.relu(self.d1(x))
        out = self.out(x)  # (batch_size, 2 * latent_layer_dim)
        return embed_text, out


class NeuronDecoder(nn.Module):
    def __init__(self, embed_layer, lstm_n_layers):
        super(NeuronDecoder, self).__init__()
        # Text
        self.embed = embed_layer
        self.n_layers = lstm_n_layers

        # Image
        self.conv1t = nn.ConvTranspose2d(32, 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.conv1 = nn.Conv2d(32, 32, kernel_size = 3, padding = "same")
        self.conv2t = nn.ConvTranspose2d(32, 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, padding = "same")
        self.conv3t = nn.ConvTranspose2d(32, 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3, padding = "same")
        self.out = nn.ConvTranspose2d(32, 3, kernel_size = 3, padding = 1)

        self.d1 = nn.Linear(latent_layer_dim, 4000)
        self.d2 = nn.Linear(4000, 4 * self.n_layers * hidden_layer_dim + img_height * img_width // 2)
        self.relu = nn.ReLU()

    def forward(self, x, start_token):
        x = self.relu(self.d1(x))
        x = self.d2(x)  # (batch_size, 8 * hidden_layer_dim + img_height * img_width / 2)
        x_text, x_image = torch.split(x, [4 * self.n_layers * hidden_layer_dim, img_height * img_width // 2], dim = 1)

        # Text
        hidden, cell = torch.split(x_text, 2 * self.n_layers * hidden_layer_dim, dim = 1)  # (batch_size, 4 * hidden_layer_dim)
        hidden = hidden.reshape(-1, 2 * self.n_layers, hidden_layer_dim).transpose(0, 1).contiguous()
        cell = cell.reshape(-1, 2 * self.n_layers, hidden_layer_dim).transpose(0, 1).contiguous()  # (4, batch_size, hidden_layer_dim)
        embed_token = self.embed(start_token)  # (batch_size, seq_len, word_embed_dim)

        # Image
        img = torch.reshape(x_image, (-1, 32, img_height // 8, img_width // 8))
        img = self.relu(self.conv1t(img))
        img = self.relu(self.conv1(img))
        img = self.relu(self.conv2t(img))
        img = self.relu(self.conv2(img))
        img = self.relu(self.conv3t(img))
        img = self.relu(self.conv3(img))
        out = self.out(img)  # (batch_size, img_height, img_width)

        return out, embed_token, (hidden, cell)


class EmbeddingDecoder(nn.Module):
    def __init__(self, lstm_n_layers):
        super(EmbeddingDecoder, self).__init__()
        self.n_layers = lstm_n_layers
        self.lstm = nn.LSTM(word_embed_dim, hidden_layer_dim, num_layers = self.n_layers, batch_first = True, bidirectional = True)
        self.out = nn.Linear(2 * hidden_layer_dim, word_embed_dim)

    def forward(self, embed_text, hidden_cell):
        output_vec, (hidden, cell) = self.lstm(embed_text, hidden_cell)  # (batch_size, 1, 2 * hidden_layer_dim)
        out = self.out(output_vec)  # (batch_size, 1, word_embed_dim)
        return out, output_vec, (hidden, cell)


def resample(mean, log_var):
    eps = torch.randn(mean.shape, device = device)
    return mean + eps * torch.exp(log_var / 2)


def get_embed_weights(vocab_size):
    if source == "cn":
        weights = torch.zeros(vocab_size, word_embed_dim)
        for i in range(vocab_size):
            token = vocab.lookup_token(i)
            if token in chn_word2vec:
                weights[i, :] = chn_word2vec[token]

    else:
        weights = glove.get_vecs_by_tokens(vocab.lookup_tokens(list(range(vocab_size))))

    return nn.Parameter(weights)


class CRVAE(nn.Module):
    def __init__(self, vocab_size, lstm_n_layers):
        super(CRVAE, self).__init__()
        self.vocab_size = vocab_size
        self.lstm_n_layers = lstm_n_layers

        self.embed = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = word_embed_dim)
        self.embed.weight.requires_grad = False
        self.embed.weight = get_embed_weights(self.vocab_size)

        self.encoder = Encoder(self.embed, self.lstm_n_layers).to(device)
        self.neuron_decoder = NeuronDecoder(self.embed, self.lstm_n_layers).to(device)
        self.embedding_decoder = EmbeddingDecoder(self.lstm_n_layers).to(device)

    def forward(self, input_text, input_image):
        # Encode
        hidden, cell = self.init_hidden_cell(input_text.shape[0])
        seq_len = input_text.shape[1]
        embed_text, latent_layer = self.encoder(input_text, input_image, (hidden, cell))

        # Resample
        latent_mean, latent_log_var = torch.split(latent_layer, latent_layer_dim, dim = 1)
        latent_layer = resample(latent_mean, latent_log_var)
        start_token = torch.tensor([sos_value] * input_text.shape[0], dtype = torch.int64).unsqueeze(1).to(device)

        # Decode image
        image_recon, embed_token, (hidden, cell) = self.neuron_decoder(latent_layer, start_token)

        # Decode text embedding
        embed_text_recon = torch.zeros(embed_text.shape, device = device)
        out, output_vec, (hidden, cell) = self.embedding_decoder(embed_token, (hidden, cell))
        embed_text_recon[:, 0, :] = out.squeeze(1)

        for i in range(seq_len - 1):
            out, output_vec, (hidden, cell) = self.embedding_decoder(embed_text[:, i, :].unsqueeze(1), (hidden, cell))
            embed_text_recon[:, i + 1, :] = out.squeeze(1)

        return latent_mean, embed_text, embed_text_recon, image_recon

    def init_hidden_cell(self, n):
        return (torch.zeros((2 * self.lstm_n_layers, n, hidden_layer_dim), device = device),
                torch.zeros((2 * self.lstm_n_layers, n, hidden_layer_dim), device = device))


# ========== Model Configuration ==========
model = CRVAE(vocab_size = len(vocab), lstm_n_layers = 2).to(device)
mse_loss = nn.MSELoss().to(device)
optim = torch.optim.Adam(model.parameters(), lr = lr)

# ========== Helper Functions ==========
subplot_num = math.ceil(math.sqrt(batch_size))


def generate_reconstructed_images(image_recon):
    images = np.transpose((image_recon * 255).detach().cpu().numpy().astype("uint8"), (0, 2, 3, 1))
    for i in range(images.shape[0]):
        plt.subplot(subplot_num, subplot_num, i + 1)
        plt.imshow(images[i, :, :, :])
        plt.axis("off")

    plt.savefig(data_save_dir + "test/epoch_{:03d}.png".format(epoch))
    plt.show()


def generate_reconstructed_texts(embed_weight, embed_text_recon):
    embed_text_flat = embed_text_recon.reshape(-1, word_embed_dim)
    decode_idx = []
    for i in range(embed_text_flat.shape[0]):
        dist = torch.linalg.norm(embed_weight - embed_text_flat[i, :], ord = 2, dim = 1)
        nearest_neighbor = torch.argmin(dist).item()
        if nearest_neighbor > 1:
            decode_idx.append(nearest_neighbor)

    decode_text = " ".join(vocab.lookup_tokens(decode_idx))
    # print(decode_text)
    return decode_text


# ========== Training Session ==========
loss_total_list, loss_text_list, loss_image_list, decode_text_list = [], [], [], []
for epoch in range(num_epochs):
    loss_total, loss_text, loss_image = (0, 0, 0)
    for idx, (text, image) in enumerate(zip(data_text_batch, data_image_batch)):
        optim.zero_grad()
        _, embed_text, embed_text_recon, image_recon = model(text, image)
        text_loss = mse_loss(embed_text, embed_text_recon)
        image_loss = mse_loss(image, image_recon)
        loss = image_loss + lambd * text_loss
        loss.backward()
        optim.step()

        loss_total += loss.item()
        loss_text += text_loss.item()
        loss_image += image_loss.item()

        if idx == 0 and epoch % 10 == 0:
            generate_reconstructed_images(image_recon)
            decode_text = generate_reconstructed_texts(model.embed.weight, embed_text_recon)
            decode_text_list.append(decode_text)

    if epoch % 10 == 0:
        print("Epoch:", epoch, "\tMSE:", loss_total, "\t(Text:", loss_text, "\tImage:", loss_image, ")")
        loss_total_list.append(loss_total)
        loss_text_list.append(loss_text)
        loss_image_list.append(loss_image)

print("Session Terminated.")

# ========== Evaluation ==========
with open(data_save_dir + "text_decode.txt", "w") as f:
    f.write("")
    print("Clearing historical text...")

with open(data_save_dir + "text_decode.txt", "a", encoding = "utf-8") as f:
    for idx, text in enumerate(decode_text_list):
        f.write("Epoch {}:\n".format(idx * 10))
        f.write(text + "\n")
        f.write("\n")

plt.plot(np.array(loss_image_list), label = "Image")
plt.plot(lambd * np.array(loss_text_list), label = "Text")
plt.plot(np.array(loss_total_list), label = "Total")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(data_save_dir + "loss.png")
plt.show()

# ========== Deployment (Output Encoded Data) ==========
latent_list = []
with torch.no_grad():
    for text, image in zip(data_text_batch, data_image_batch):
        latent, _, _, _ = model(text, image)
        latent_list.append(latent)

latent_vec = torch.cat(latent_list, dim = 0)
print(latent_vec.shape)
np.savetxt(data_save_dir + "data.csv", latent_vec.cpu().numpy(), delimiter = ",")
