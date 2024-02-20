import torch
from transformers import MusicGenTokenizer, MusicGenForConditionalGeneration

# Load the pre-trained MusicGen model and tokenizer
model_name = "facebook/musicgen-large"
tokenizer = MusicGenTokenizer.from_pretrained(model_name)
model = MusicGenForConditionalGeneration.from_pretrained(model_name)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the custom handler function
def generate_music(prompt, melody=None):
    # Tokenize the input prompt and optional melody
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    if melody:
        melody_inputs = tokenizer(melody, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], melody_inputs["input_ids"]), dim=1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], melody_inputs["attention_mask"]), dim=1)

    # Generate music based on the inputs
    with torch.no_grad():
        output = model.generate(**inputs)

    # Decode the generated music tokens back into audio format
    generated_music = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_music

# Example usage:
if __name__ == "__main__":
    # Example input prompt and melody (optional)
    prompt = "A relaxing melody with piano"
    melody = "C D E F G A B"

    # Generate music based on the input prompt and melody
    generated_music = generate_music(prompt, melody)
    print(generated_music)
