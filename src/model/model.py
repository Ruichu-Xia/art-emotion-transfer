import torch.nn as nn

class EmotionEmbeddingModel(nn.Module):
    def __init__(self, base_model, num_classes, hidden_size, num_hidden_layers=1, freeze_layers=False):
        super(EmotionEmbeddingModel, self).__init__()
        self.base = base_model
        if freeze_layers:
            for param in self.base.parameters():
                param.requires_grad = False
                
        layers = []
        input_size = 2048  # ResNet output size

        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.2)])
            input_size = hidden_size

        self.fc_embed = nn.Sequential(*layers)

        self.fc_classify = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embeddings = self.embed(x)
        logits = self.classify(embeddings)
        return embeddings, logits

    def embed(self, x):
        x = self.base(x).pooler_output  # Extract features
        x = x.view(x.size(0), -1)  # Flatten features
        return self.fc_embed(x)

    def classify(self, x):
        return self.fc_classify(x)