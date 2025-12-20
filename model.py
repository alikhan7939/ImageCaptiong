import torch 
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        
        # حذف classifier اصلی و جایگزینی
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        
        # فریز کردن ویژگی‌گیرها
        for name, param in self.inception.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = train_CNN
                
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features))

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # بهتره batch_first=True باشه
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # features: (batch_size, embed_size)
        # captions: (batch_size, seq_len)
        
        embeddings = self.dropout(self.embed(captions[:, :-1]))  # حذف <end> برای teacher forcing
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # (batch, seq_len+1, embed_size)
        
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, features, max_length=20, start_token=1):  # start_token معمولاً index کلمه <start>
        """
        تولید کپشن به صورت greedy
        """
        self.eval()
        with torch.no_grad():
            inputs = features.unsqueeze(1)  # (batch, 1, embed_size)
            hidden = None
            caption = []
            
            for _ in range(max_length):
                lstm_out, hidden = self.lstm(inputs, hidden)
                output = self.linear(lstm_out.squeeze(1))
                predicted = output.argmax(1)
                
                caption.append(predicted.item())
                
                if predicted.item() == 2:  # فرض <end> index=2 باشه
                    break
                    
                inputs = self.embed(predicted).unsqueeze(1)
                
            return caption

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]    