from flask import Flask, request, jsonify
from PIL import Image
import torch
import base64
import io
import re
import json
import time
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, Accelerator

import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence, List, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms

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

# Initialize model and tokenizer (outside endpoint for reuse)
print("Initializing tokenizer...")
text_tokenizer, image_padding_tokens = get_tokenizer('./Language_files')

print("Loading model...")
model = MultiLLaMAForCausalLM(lang_model_path='./Language_files')

print("Loading checkpoint...")
ckpt = torch.load('./Language_files/pytorch_model.bin', map_location='cpu')

# Clean checkpoint keys
model_keys = set(model.state_dict().keys())
ckpt_keys = set(ckpt.keys())
unexpected_keys = ckpt_keys - model_keys
for key in unexpected_keys:
    del ckpt[key]

model.load_state_dict(ckpt)
model.eval()
print("Model loaded and in eval mode")



def preprocess_base64_image(base64_string):
    '''
    Convert base64 image to tensor - MATCHING ORIGINAL PREPROCESSING
    '''
    transform = transforms.Compose([                        
        transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Apply transform to get tensor of shape [3, 512, 512]
    image_tensor = transform(image)
    
    # Add batch and depth dimensions as in original: unsqueeze(0).unsqueeze(-1)
    # Result: [1, 3, 512, 512, 1]
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(-1)
    
    # Resize to add depth dimension: [1, 3, 512, 512, 4]
    target_H = 512 
    target_W = 512 
    target_D = 4
    image_tensor = torch.nn.functional.interpolate(image_tensor, size=(target_H, target_W, target_D))
    
    print(f"DEBUG: Preprocessed image shape: {image_tensor.shape}")  # Should be [1, 3, 512, 512, 4]
    return image_tensor



#

#

def combine_and_preprocess_openai(user_question, images_data, image_padding_tokens):
    '''
    Process OpenAI-style request with base64 images
    user_question: JUST the text from the last user message
    '''
    images = []
    
    print(f"DEBUG: Processing {len(images_data)} images")
    print(f"DEBUG: User question: {user_question[:200]}...")
    
    for i, img_data in enumerate(images_data):
        if i >= len(image_padding_tokens):
            break
            
        # Process base64 image
        if 'image_url' in img_data:
            data_url = img_data['image_url']['url']
            if data_url.startswith('data:image'):
                base64_str = data_url.split(',')[1]
            else:
                base64_str = data_url
        elif 'base64' in img_data:
            base64_str = img_data['base64']
        else:
            continue
            
        image_tensor = preprocess_base64_image(base64_str)
        images.append(image_tensor)
    
    # Stack images
    if images:
        vision_x = torch.cat(images, dim=0)
        vision_x = vision_x.unsqueeze(0)
    else:
        vision_x = torch.zeros((1, 0, 3, 512, 512, 4))
    
    # Insert image tag at beginning of user question
    # Format: <image>tokens</image>User question text
    text = f"<image>{image_padding_tokens[0]}</image>" + user_question if images else user_question
    
    print(f"DEBUG: Final text (first 300 chars): {text[:300]}")
    print(f"DEBUG: vision_x shape: {vision_x.shape}")
    
    return text, vision_x


#

def process_openai_messages(messages):
    '''
    Convert OpenAI-style messages to text format for model
    Returns ONLY the last user message text (where image should be inserted)
    '''
    # Find the last user message (current query)
    last_user_text = ""
    
    for message in reversed(messages):
        if message.get('role') == 'user':
            content = message.get('content', '')
            
            if isinstance(content, list):
                # Extract text parts from multimodal content
                text_parts = []
                for item in content:
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                last_user_text = ' '.join(text_parts)
            else:
                last_user_text = str(content)
            break
    print(last_user_text)
    
    return last_user_text


#


def extract_images_from_messages(messages):
    '''
    Extract base64 images from OpenAI-style messages
    Only extract from the LAST user message (current query)
    '''
    images_data = []
    
    # Only look at the last message (current query)
    if messages and messages[-1].get('role') == 'user':
        content = messages[-1].get('content', '')
        
        if isinstance(content, list):
            for item in content:
                if item.get('type') == 'image_url':
                    images_data.append(item)
    print("number images ", len(images_data))
    #print("sample  :", images_data[0])
    return images_data

def parse_model_response_to_json(model_response):
    """
    Convert model's text response to required JSON format
    Input: "Diagnosis:\n- Pneumonia: no\n- Pulmonary Edema: yes..."
    Output: {"diagnosis_flags": {"pneumonia": false, "pulmonary_edema": true...}, ...}
    """
    result = {
        "proof_of_thought": "",
        "image_findings": [],
        "diagnosis_flags": {},
        "acute_abnormality": False,
        "confidence_score": 0,
        "summary": ""
    }
    
    # Extract image findings from "Findings:" section
    if "Findings:" in model_response:
        findings_start = model_response.find("Findings:")
        findings_end = model_response.find("Critique:") if "Critique:" in model_response else model_response.find("Diagnosis:")
        if findings_end > findings_start:
            findings_text = model_response[findings_start:findings_end]
            # Extract bullet points like "- Cardiac Shadow"
            import re
            bullet_points = re.findall(r'-\s*(.+)', findings_text)
            result["image_findings"] = bullet_points
    
    # Extract diagnosis flags from "Diagnosis:" section
    if "Diagnosis:" in model_response:
        diag_start = model_response.find("Diagnosis:")
        diag_text = model_response[diag_start:]
        
        # Parse lines like "- Pneumonia: yes/no"
        import re
        diag_lines = re.findall(r'-\s*([^:]+):\s*(yes|no)', diag_text, re.IGNORECASE)
        
        for pathology, value in diag_lines:
            # Clean pathology name
            clean_path = pathology.strip().lower().replace(" ", "_")
            # Convert yes/no to boolean
            result["diagnosis_flags"][clean_path] = value.lower() == "yes"
    
    # Extract proof of thought
    if "Observation:" in model_response and "Analyze:" in model_response:
        obs_start = model_response.find("Observation:")
        analyze_start = model_response.find("Analyze:")
        if analyze_start > obs_start:
            result["proof_of_thought"] = model_response[obs_start:analyze_start].strip()
    
    return result

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    '''
    OpenAI-compatible endpoint
    '''
    try:
        data = request.json
        
        # Extract parameters
        model_name = data.get('model', 'radfm-vision')
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 400)  # Max new tokens for generation
        response_format = data.get('response_format', {})
        
        idx=1
        for  m in messages:
            print("message ",  idx)
            idx+=1
            for key, value in m.items():
                print("key ", key)
        # Process messages
        text_prompt = process_openai_messages(messages)
        images_data = extract_images_from_messages(messages)
        
        # Combine text and images
        text, vision_x = combine_and_preprocess_openai(text_prompt, images_data, image_padding_tokens)

        print(f"DEBUG: Text length: {len(text)}")
        print(f"DEBUG: Contains image placeholder? {'<image>' in text}")
        print(f"DEBUG: First 300 chars of text: {text[:300]}")
        
        # Tokenize text - your model expects lang_x (token IDs)
        text_tokens = text_tokenizer(text, return_tensors="pt")
        lang_x = text_tokens['input_ids']

        print("DEBUG: tokenizer done")
        
        # Generate response using your model's generate() method
        with torch.no_grad():

            #clear cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            model.eval()

            # Your model's generate() expects lang_x and vision_x
            generated_tokens = model.generate(
                lang_x=lang_x,
                vision_x=vision_x
            )
            
            print("DEBUG: Done generation")
            # Decode response
            response_text = text_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

            print(f"DEBUG: num generated tokens: {len(generated_tokens[0])}")
            

            print(f"DEBUG: Raw model response: {response_text[:1000]}")
        
        # Extract only the assistant's response (remove the prompt)
        if "Assistant:" in response_text:
            # Get everything after the last "Assistant:"
            parts = response_text.split("Assistant:")
            response_text = parts[-1].strip()
        elif "User:" in response_text:
            # If no Assistant tag, get everything after the prompt
            parts = response_text.split(text_prompt)
            if len(parts) > 1:
                response_text = parts[-1].strip()
        
        # Format response in OpenAI style
        if response_format.get('type') == 'json_object':
            # Parse the model's text response into required JSON format
            parsed_json = parse_model_response_to_json(response_text)
            response_text = json.dumps(parsed_json)
        
        # Calculate token counts
        prompt_tokens = len(lang_x[0])
        completion_tokens = len(generated_tokens[0]) - prompt_tokens
        
        # Create OpenAI-style response
        response = {
            "id": f"chatcmpl-{int(time.time())}{torch.randint(1000, 10000, (1,)).item()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

        print(f"DEBUG: sending respones : {response}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        return jsonify({
            "error": {
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        }), 500

@app.route('/infer', methods=['POST'])
def infer():
    '''
    Legacy endpoint (for backward compatibility)
    '''
    try:
        data = request.json
        question = data.get('question', '')
        images = data.get('images', [])
        
        # Convert to messages format
        messages = [
            {"role": "user", "content": question}
        ]
        
        # Add images if provided
        if images:
            content = [{"type": "text", "text": question}]
            for img in images:
                if 'base64' in img:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img['base64']}
                    })
            messages[0]['content'] = content
        
        # Add a system message for better responses
        system_message = {
            "role": "system",
            "content": "You are a helpful medical AI assistant. Provide clear, accurate responses."
        }
        messages.insert(0, system_message)
        
        # Use chat completion endpoint
        return chat_completions()
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    '''
    OpenAI-compatible models endpoint
    '''
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "radfm-vision",
                "object": "model",
                "created": 1686935000,
                "owned_by": "radfm",
                "permission": [],
                "root": "radfm-vision",
                "parent": None
            }
        ]
    })

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "radfm-vision",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/infer"]
    })

if __name__ == '__main__':
    print("Starting RadFM Vision Server...")
    print("Available endpoints:")
    print("  - GET  /                Health check")
    print("  - POST /v1/chat/completions  OpenAI-compatible chat")
    print("  - GET  /v1/models       List models")
    print("  - POST /infer           Legacy endpoint")
    app.run(host='0.0.0.0', port=8000, debug=False)
