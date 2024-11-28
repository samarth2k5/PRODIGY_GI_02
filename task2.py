from diffusers import StableDiffusionPipeline
import torch

# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)  # Remove torch_dtype argument

# Set to CPU
pipe = pipe.to("cpu")  # Ensure it uses CPU

# Text description of the image
description = "A futuristic cityscape with flying cars"

# Generate the image
image = pipe(description).images[0]

# Save the image
image.save("output.png")

print("Image created and saved as output.png")
