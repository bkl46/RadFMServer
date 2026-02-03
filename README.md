download checkpoint https://huggingface.co/datasets/chaoyi-wu/RadFM_data_csv and store pytorch_model.bin in /Language_files

this server is meant to run locally and thus should have access to all images paths fed to it. When hitting the /v1/chat/completions endpoint, pass in image paths relative to the directory you are running the server from. 

run 
pip install -r requirements.txt
python radfmserver.py

server will be running on localhost port 8000 by default
