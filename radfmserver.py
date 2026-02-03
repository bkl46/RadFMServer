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
print("Model loaded and in eval mode")

model.to('cuda')

model.eval()



#

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

    print("DEBUG: attempting to combine and preprocess")
    # Define image transformation pipeline
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    
    images = []
    new_qestions = [_ for _ in question]  # Convert question string to list of characters
    padding_index = 0

    print("DEBUG: transform made and prompt converted to chars")
    

    print("DEBUG: lenght of  image list ", len(image_list))
    idx =0
    # Process each image in the list
    for img in image_list:
        print(f"DEBUG for idx: {idx}")
        img_path = img['img_path']
        position = img['position']  # Where to insert the image in the text

        print(f"DEBUG: path: {img_path} and position {position}")
        
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
        print("DEBUG: right before insertiing tokesn") 

        
        print(f"DEBUG INSERT: At position={position}, inserting in reverse order")
        print(f"DEBUG INSERT: List length before: {len(new_qestions)}")
        print(f"DEBUG INSERT: Character at position before: '{new_qestions[position] if position < len(new_qestions) else 'END'}'")
        
        # CORRECT ORDER: Insert in REVERSE order so they appear in FORWARD order
        new_qestions.insert(position, "</image>")
        print(f"DEBUG INSERT: After 1st insert, pos {position} now has: '{new_qestions[position]}'")
        
        new_qestions.insert(position, image_padding_tokens[padding_index])
        print(f"DEBUG INSERT: After 2nd insert, pos {position} now has: '{new_qestions[position][:20]}...'")
        
        new_qestions.insert(position, "<image>")
        print(f"DEBUG INSERT: After 3rd insert, pos {position} now has: '{new_qestions[position]}'")
        print(f"DEBUG INSERT: Next few chars: {''.join(new_qestions[position:position+5])}")

        padding_index += 1
    
    # Stack all images into a batch and add batch dimension
    vision_x = torch.cat(images, dim=1).unsqueeze(0)  # Cat tensors and expand the batch_size dim


    print(f"ORIGINAL DEBUG: vision x shape after at {torch.cat(images, dim=1).shape}")
    print(f"ORIGINAL DEBUG: vision x shape after unsqueeze: {vision_x.shape}")
    
    # Join the character list back into a string
    text = ''.join(new_qestions) 
    return text, vision_x
    



#test vision encoder
def test_vision_encoder_is_working():
    """
    Extreme diagnostic test. 
    Prompt forces the model to describe ONLY what it sees in the image.
    """
    print("\n" + "="*60)
    print("VISION-ONLY DIAGNOSTIC TEST")
    print("="*60)
    
    # ULTRA-SIMPLE PROMPT: Just describe the image, no medical jargon, no structure.
    diagnostic_prompt = "This is a picture of:"
    
    test_image = [{
        'img_path': '/home/bkl46/M/MedVLM001/data/images/images_normalized/612_IM-2199-2001.dcm.png',
        'position': len("diagnostic_prompt")  # Insert image token AFTER the prompt text
    }]
    
    # 1. Process through combine_and_preprocess
    text, vision_x = combine_and_preprocess(diagnostic_prompt, test_image, image_padding_tokens)
    print(f"DEBUG: Processed text starts with: {text[:150]}")
    print(f"DEBUG: vision_x shape: {vision_x.shape}")
    
    # 2. Tokenize and move to GPU
    lang_x = text_tokenizer(
        text, max_length=2048, truncation=True, return_tensors="pt"
    )['input_ids']

    print(f"DEBUG: lang_x shape: {lang_x.shape}")
    
    # 3. Generate with VERY permissive settings
    print("DEBUG: Generating...")
    with torch.no_grad():
        generated_tokens = model.generate(
            lang_x=lang_x,
            vision_x=vision_x,
            )
    
    response = text_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"\nDIAGNOSTIC RESULT:")
    print(f"Prompt: '{diagnostic_prompt}'")
    print(f"Full Model Output: '{response}'")
    
    # 4. CRITICAL: Strip the original prompt to see ONLY what the model generated.
    generated_part = response[len(diagnostic_prompt):].strip()
    print(f"Model's New Generation: '{generated_part}'")
    print("="*60)
    
    # 5. Analyze the result
    if len(generated_part) == 0:
        print("❌ ALERT: Model generated NOTHING. Vision path may be completely broken.")
    elif "picture" in generated_part.lower() or "image" in generated_part.lower() or "photo" in generated_part.lower():
        print("⚠️  Warning: Model is parroting generic words about images, not describing content.")
    else:
        print("✅ Model generated *some* descriptive text. Vision path is likely active.")
    return generated_part



    #### create query

def create_radfm_medical_qa(bg_text: str, image_paths: List[str], target_pathologies: List[str]) -> tuple:
    """
    Create a RadFM-optimized medical Q&A prompt that mimics the demo style.
    
    Args:
        bg_text: Clinical background
        image_paths: List of image file paths
        target_pathologies: List of pathologies to query
    
    Returns:
        Tuple of (prompt_text, image_list) ready for combine_and_preprocess
    """
    # Build the Q&A prompt in the exact style of the working demo
    # Place the image reference WHERE it makes sense in the question
    base_prompt = f"Background: {bg_text}\n"
    
    # We'll ask one pathology at a time for clarity
    # For multiple pathologies, we could loop or ask combined questions
    pathology = target_pathologies[0] if target_pathologies else ""
    
    # DIRECT QUESTION format (like the demo)
    question = f"Can you identify any visible signs of {pathology} in the image?"
    
    # Combine background and question
    full_prompt = base_prompt + question
    
    # Prepare image list - insert image BEFORE "in the image?" part
    # Find where to insert: before "in the image?"
    insert_position = full_prompt.find("in the image?")
    if insert_position == -1:
        insert_position = len(full_prompt)  # Fallback to end
    
    image_list = []
    for i, img_path in enumerate(image_paths):
        image_list.append({
            'img_path': img_path,
            'position': insert_position
        })
    
    return full_prompt, image_list

# create query: Batch multiple pathologies in one prompt
def create_radfm_batch_qa(bg_text: str, image_paths: List[str], target_pathologies: List[str]) -> tuple:
    """
    Ask about multiple pathologies in a single prompt.
    """
    base_prompt = f"Background: {bg_text}\n"
    questions = []
    
    for pathology in target_pathologies:
        questions.append(f"Is {pathology} visible in the image?")
    
    full_prompt = base_prompt + " ".join(questions)
    
    # Insert image before the first "in the image" reference
    insert_position = full_prompt.find("in the image")
    if insert_position == -1:
        insert_position = len(full_prompt)
    
    image_list = []
    for img_path in image_paths:
        image_list.append({
            'img_path': img_path,
            'position': insert_position
        })
    
    return full_prompt, image_list


def parse_detailed_findings(response: str, target_findings: List[str]) -> Dict:
    """
    Parse detailed radiology report for target findings with better logic.
    """
    response_lower = response.lower()
    findings = {}
    
    # Map alternative/synonym terms to your target findings
    synonym_map = {
        "pneumonia": ["pneumonia", "consolidation", "infiltrate", "opacity"],
        "pleural_effusion": ["pleural effusion", "effusion", "fluid"],
        "atelectasis": ["atelectasis", "collapse", "volume loss"],
        "cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
        "pneumothorax": ["pneumothorax", "collapsed lung"],
        "pulmonary_edema": ["pulmonary edema", "edema", "fluid overload"],
        # Add more mappings...
    }
    
    for target in target_findings:
        found = False
        # Check target name
        if target.lower() in response_lower:
            found = True
        # Check synonyms
        elif target in synonym_map:
            for synonym in synonym_map[target]:
                if synonym in response_lower:
                    found = True
                    break
        
        # Check for NEGATIONS (e.g., "No pulmonary nodules")
        if found:
            # Look for negations before the finding
            words = response_lower.split()
            target_idx = response_lower.find(target.lower())
            if target_idx > -1:
                # Check preceding words for negations
                preceding_text = response_lower[max(0, target_idx-50):target_idx]
                if any(neg in preceding_text for neg in ["no ", "without ", "absence of ", "negative for "]):
                    found = False
        
        findings[target] = found
    
    return findings


#process query given the input messages content
def process_single_query_qa(sample_data: Dict[str, any]) -> Dict[str, any]:
    """
    Query all pathologies in a single prompt (more efficient).
    """
    bg_text = sample_data.get('background', '')
    image_paths = sample_data.get('images', [])
    
    DEFAULT_FINDINGS = ["pneumonia", "pneumothorax", "pulmonary_edema", "consolidation", 
                       "atelectasis", "fracture", "pleural_effusion", "cardiomegaly", 
                       "emphysema", "fibrosis"]
    
    # Create a comprehensive single question
    findings_list = ", ".join(DEFAULT_FINDINGS)
    prompt = (
        f"Background: {bg_text}\n"
        f"Examine this chest radiograph and indicate which findings are present from this list: {findings_list}.\n"
        f"Respond with only the names of the findings that are visible in the image."
    )
    
    # Insert image at a logical point
    images = [{
        'img_path': image_paths[0] if image_paths else '',
        'position': prompt.find("in the image.")
    }]
    
    # Process through model
    text, vision_x = combine_and_preprocess(prompt, images, image_padding_tokens)

    print("DEBUG: text ", text)
    lang_x = text_tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")['input_ids'].to('cuda')

    vision_x = vision_x.to('cuda')
    
    with torch.no_grad():
        generated_tokens = model.generate(
            lang_x=lang_x,
            vision_x=vision_x
        )
    
    response = text_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print("DEBUG: raw response: ", response)

    
    # Parse the response
    result = {
        "proof_of_thought": f"Single comprehensive analysis: {response}",
        "image_findings": [],
        "diagnosis_flags": {},
        "acute_abnormality": False,
        "confidence_score": 70,
        "summary": response[:200]  # Use first 200 chars as summary
    }
    
    # Check which pathologies are mentioned
    response_lower = response.lower()
    for pathology in DEFAULT_FINDINGS:
        if pathology.lower() in response_lower:
            result["diagnosis_flags"][pathology] = True
            result["image_findings"].append(pathology)
            result["acute_abnormality"] = True
        else:
            result["diagnosis_flags"][pathology] = False
    
    return result



def test_final_pipeline():
    """
    Test the complete medical Q&A pipeline.
    """
    test_data = {
        'background': 'INDICATION: 65-year-old woman with chest pain.',
        'images': ['/home/bkl46/M/MedVLM001/data/images/images_normalized/612_IM-2199-2001.dcm.png']
    }

    print("\n" + "="*60)
    print("FINAL PIPELINE TEST")
    print("="*60)

    result = process_single_query_qa(test_data)

    print(f"\nFinal Result:")
    print(json.dumps(result, indent=2))
    print("="*60)

    return result





@app.route('/tpipe', methods=['GET'])
def runtest():
    result = test_final_pipeline()
    return {"test": result}

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    '''
    OpenAI-compatible endpoint
    '''
    try:
        data = request.json
        
        # Extract parameters
        model_name = 'radfm-vision' #data.get('model', 'radfm-vision')
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 400)  # Max new tokens for generation
        response_format = data.get('response_format', {})
        

        print("DEBUG: recieving message, size: ", len(messages))


        background = messages[0].get("background")
        paths = message[1].get("image")

        data = {
            'background': background,
            'images':paths 
        }


        result = process_single_query_qa(data)

       
        print("DEBUG: outsdie generation")
       

        print(f"DEBUG: sending respones : {result}")
        
        return jsonify(result)
        
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
