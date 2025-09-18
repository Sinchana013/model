import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import threading
import sounddevice as sd

# ---------------- CONFIG ----------------
FRAME_SKIP = 5
SEQ_LEN = 12
THRESHOLD = 0.75
NUM_STEPS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Beep function ----------------
def beep():
    fs = 44100
    duration = 0.5
    f = 1000
    t = np.linspace(0, duration, int(fs*duration), False)
    tone = np.sin(2 * np.pi * f * t)
    sd.play(tone, fs)
    sd.wait()

# ---------------- Model Definitions ----------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]  # remove FC layer
        self.feature_extractor = nn.Sequential(*modules)

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            return features.view(features.size(0), -1)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=1, num_classes=128):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ---------------- Load Models ----------------
cnn_model = CNNFeatureExtractor().to(device)
cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
cnn_model.eval()

lstm_model = LSTMClassifier().to(device)
lstm_model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
lstm_model.eval()

# ---------------- Load Reference Features ----------------
reference_features = np.load("reference_features.npy")

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- Live Loop ----------------
cap = cv2.VideoCapture(0)
frame_buffer = []
frame_count = 0
current_step = 0
detected_steps = []
missed_flag = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
        continue

    # Preprocess frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)

    # CNN feature extraction
    with torch.no_grad():
        cnn_feat = cnn_model(img).squeeze().cpu().numpy()
    frame_buffer.append(cnn_feat)

    # Keep only last SEQ_LEN frames
    if len(frame_buffer) > SEQ_LEN:
        frame_buffer.pop(0)

    if len(frame_buffer) == SEQ_LEN:
        seq_input = torch.tensor(np.array(frame_buffer), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            live_feat = lstm_model(seq_input).cpu().numpy().reshape(-1)

        # Cosine similarity with reference steps
        sims = cosine_similarity(live_feat.reshape(1, -1), reference_features)
        best_step = sims.argmax()
        max_sim = sims.max()

        status = None
        if max_sim >= THRESHOLD:
            if best_step == current_step:
                status = f"✅ Step {best_step+1} detected"
                detected_steps.append(best_step)
                current_step += 1 if current_step < NUM_STEPS-1 else 0
                missed_flag = False
            elif best_step > current_step:
                if not missed_flag:
                    status = f"❌ Missed Step {current_step+1}, Detected Step {best_step+1}"
                    threading.Thread(target=beep).start()
                    detected_steps.append(best_step)
                    current_step = best_step + 1 if best_step < NUM_STEPS-1 else 0
                    missed_flag = True
            else:
                status = f"✅ Step {best_step+1} detected"

            if status:
                color = (0, 255, 0) if "✅" in status else (0, 0, 255)
                cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                print(f"[DEBUG] {status}, Similarity: {max_sim:.3f}")

    cv2.imshow("Live", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
