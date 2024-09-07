from flask import Flask, request, send_file, render_template
import torch
from PIL import Image
import io
import numpy as np
import torch.nn as nn


class ColorAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 64, 3, stride=2)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(128, 3, 3, stride=2, output_padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        d4 = self.relu(self.down4(d3))

        u1 = self.relu(self.up1(d4))

        min_height = min(u1.size(2), d3.size(2))
        min_width = min(u1.size(3), d3.size(3))
        u1_cropped = u1[:, :, :min_height, :min_width]
        d3_cropped = d3[:, :, :min_height, :min_width]
        u2 = self.relu(self.up2(torch.cat((u1_cropped, d3_cropped), dim=1)))

        min_height = min(u2.size(2), d2.size(2))
        min_width = min(u2.size(3), d2.size(3))
        u2_cropped = u2[:, :, :min_height, :min_width]
        d2_cropped = d2[:, :, :min_height, :min_width]
        u3 = self.relu(self.up3(torch.cat((u2_cropped, d2_cropped), dim=1)))

        min_height = min(u3.size(2), d1.size(2))
        min_width = min(u3.size(3), d1.size(3))
        u3_cropped = u3[:, :, :min_height, :min_width]
        d1_cropped = d1[:, :, :min_height, :min_width]
        u4 = self.sigmoid(self.up4(torch.cat((u3_cropped, d1_cropped), dim=1)))

        return u4


model = ColorAutoEncoder()


model.load_state_dict(
    torch.load(
        "models/color_autoencoder.pth",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
)


model.eval()


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("button-change.html")


@app.route("/change", methods=["GET", "POST"])
def change():
    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file).convert("L")

        img_tensor = (
            torch.from_numpy(np.array(img)).unsqueeze(0).unsqueeze(0).float() / 255.0
        )  # Adjust dimensions

        with torch.no_grad():
            colorized_tensor = model(img_tensor)

        colorized_image = (
            (colorized_tensor.squeeze(0).permute(1, 2, 0) * 255).byte().numpy()
        )
        colorized_image = Image.fromarray(colorized_image)

        img_io = io.BytesIO()
        colorized_image.save(img_io, "JPEG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/jpeg")

    return render_template("panel.html")


if __name__ == "__main__":
    app.run(debug=True)
