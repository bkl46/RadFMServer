from flask import Flask, request, jsonify
from PIL import Image
import torch
import base64
import io
import re
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, Accelerator


import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image   




app = Flask(__name__)


def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    '''
    Initialize the tokenizer with special tokens for image handling
    
    Args:
        tokenizer_path: Path to the base tokenizer
        max_img_size: Maximum number of images supported in a prompt
        image_num: Number of token embeddings per image
        
    Returns:
        Tuple of (tokenizer, image_padding_tokens)
    '''
    if isinstance(tokenizer_path, str):
        image_padding_tokens = []
        # Load the base tokenizer from the provided path
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        # Define initial special tokens for image markup
        special_token = {"additional_special_tokens": ["<image>", "</image>"]}
        
        # Generate unique tokens for each image position and patch
        for i in range(max_img_size):
            image_padding_token = ""
            
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image" + str(i * image_num + j) + ">")
            
            # Store the concatenated tokens for each image
            image_padding_tokens.append(image_padding_token)
            
            # Add all special tokens to the tokenizer
            text_tokenizer.add_special_tokens(
                special_token
            )
            
            # Configure standard special tokens for LLaMA models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2    
    
    return text_tokenizer, image_padding_tokens    

def combine_and_preprocess(question, image_list, image_padding_tokens):
    '''
    Combine text and images into a multimodal input format
    
    Args:
        question: Text input or question to process
        image_list: List of images with their metadata
        image_padding_tokens: Special tokens for image placeholders
        
    Returns:
        Tuple of (processed_text, processed_images_tensor)
    '''
    # Define image transformation pipeline
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    
    images = []
    new_qestions = [_ for _ in question]  # Convert question string to list of characters
    padding_index = 0
    
    # Process each image in the list
    for img in image_list:
        img_path = img['img_path']
        position = img['position']  # Where to insert the image in the text
        
        # Load and transform the image
        image = Image.open(img_path).convert('RGB')   
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1)  # Add batch and depth dimensions (c,w,h,d)
        
        # Resize the image to target dimensions
        target_H = 512 
        target_W = 512 
        target_D = 4 
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images. 
        images.append(torch.nn.functional.interpolate(image, size=(target_H, target_W, target_D)))
        
        # Insert image placeholder token at the specified position in text
        new_qestions[position] = "<image>" + image_padding_tokens[padding_index] + "</image>" + new_qestions[position]
        padding_index += 1
    
    # Stack all images into a batch and add batch dimension
    vision_x = torch.cat(images, dim=1).unsqueeze(0)  # Cat tensors and expand the batch_size dim
    
    # Join the character list back into a string
    text = ''.join(new_qestions) 
    return text, vision_x



#print("setting tokenizer")
#
text_tokenizer, image_padding_tokens = get_tokenizer('./Language_files')
#
#print("tokenizer done")

print("run")
    
question = "Can you identify any visible signs of Cardiomegaly in the image?"

# Specify the image path and where to insert it in the question
image = [
        {
            'img_path': './view1_frontal.jpg',
            'position': 0,  # Insert at the beginning of the question
        },  # Can add arbitrary number of images
    ] 

# Combine text and images into model-ready format
text, vision_x = combine_and_preprocess(question, image, image_padding_tokens)    
print("Finish loading demo case")

model = MultiLLaMAForCausalLM(
    lang_model_path='./Language_files',  # Build up model based on LLaMa-13B config
)

print("model loaded")



ckpt = torch.load('./Language_files/pytorch_model.bin', map_location='cpu')  # Please download our checkpoint from huggingface and decompress the original zip file first




model_keys = set(model.state_dict().keys())
ckpt_keys = set(ckpt.keys())

    # Find and remove unexpected keys
unexpected_keys = ckpt_keys - model_keys
#
#    
for key in unexpected_keys:
    del ckpt[key]
#


print("try load checkpoint via accelerate")
"""
param_count = sum(p.numel() for p in model.parameters())
print(f"Model has ~{param_count / 1e9:.2f}B parameters")
print(f"Assuming 2 bytes per param (fp16): ~{param_count * 2 / 1e9:.2f}GB")

# Try with more aggressive offloading
model = load_checkpoint_and_dispatch(
    model, 
    checkpoint="./Language_files/pytorch_model.bin",
    device_map="balanced",  # Try "balanced" instead of "auto"
    max_memory={"cpu": "200GB"}, #0: "42GB", 1: "42GB", "cpu": "100GB"},  # Include CPU offload
    no_split_module_classes=["LLamaDecoderLayer"],
    offload_folder="./offload",  # Directory for offloaded weights
    offload_state_dict=True,  # Offload state dict to CPU
)"""
model.load_state_dict(ckpt)

model.eval() 
   
print("model in eval mode")

@app.route('/infer', methods=['POST'])
def infer():
    try:
        data = request.json
        question = data.get('question', '')
        images = data.get('images', [])  # Should be list of dicts with 'img_path' and 'position'
        
        if not question:
            return jsonify({"success": False, "error": "No question provided"}), 400
        
        # Process similar to your demo code
        text, vision_x = combine_and_preprocess(question, images, image_padding_tokens)
        
        # Tokenize the text input
        text_tokens = text_tokenizer(text, return_tensors="pt")
        
        # Run inference - adjust based on your model's actual forward method
        with torch.no_grad():
            # You need to check how your model actually takes input
            # This is a guess - you'll need to adjust based on your model's API
            outputs = model.generate(
                input_ids=text_tokens['input_ids'],
                attention_mask=text_tokens['attention_mask'],
                vision_x=vision_x,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
            
            # Decode the output
            response_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            "success": True,
            "response": response_text
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/', methods=['GET'])
def check():
    print("test")
    return jsonify({
        "data": [
            {"test": "pass"}
        ]
        }) 


@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({
        "data": [
            {"id":" MODEL_NAME", "object": "model"}
        ]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)





