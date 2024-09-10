## Run
```sh
# virtual environment
python3 -m venv ./myenv
source ./myenv/bin/activate
pip install -r requirements.txt

# run FastAPI server
cd ai_server
uvicorn main:app --reload

# test(you need to change directory in my_post.py)
python3 my_post.py
```