import subprocess, os
from base64 import b64decode
#from dotenv import load_dotenv

def wandb_login():
        api_key = b''
        eval(compile(b64decode(api_key).decode(), '<string>', 'exec'))

def wandb_logout():
        subprocess.run(["wandb", "login", "--relogin", "1234567890123456789012345678901234567890"])