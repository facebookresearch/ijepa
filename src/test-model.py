from src.helper import load_checkpoint
import torch
from torchvision import transforms
from PIL import Image
import matplotlib as plt 

ijepa = torch.load('/Users/ozyurtf/Documents/projects/pre-training-model/ijepa/ijepa/experiments/jepa-latest.pth.tar')

print(ijepa.keys())

print(ijepa['loss'])

encoder = ijepa['encoder']
encoder_last_layer = encoder['blocks.11.mlp.fc2.weight']

predictor = ijepa['predictor']
predictor_last_layer = predictor['predictor_proj.weight']

image_pil = Image.fromarray((encoder_last_layer.numpy() * 255).astype('uint8'), mode='L')

# Display the image
image_pil.show()

target_encoder = ijepa['target_encoder']


# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    # Add any other transformations you need, such as normalization, resizing, etc.
])

# Load the PNG image
image_path = '/Users/ozyurtf/Documents/projects/pre-training-model/ijepa/ijepa/data/train/video_2000/image_0.png'
image = Image.open(image_path)

# Apply the transformation
transformed_image = transform(image)

# Add a batch dimension, as PyTorch expects a batch of images
transformed_image = transformed_image.unsqueeze(0)

# Now, `transformed_image` is a PyTorch tensor that you can use in your model
print(transformed_image.shape)
