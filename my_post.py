import requests

def send_frames_batch(frame_folder="", frame_count=50):
    url = "http://127.0.0.1:8000/inference"
      
    # 여러 파일을 한 번에 전송할 수 있도록 파일 리스트 생성
    files = []
    for idx in range(frame_count):
        filename = f"{frame_folder}/{idx}.png"
        with open(filename, 'rb') as file:
            # append 대신 딕셔너리에서 key-value로 여러 파일 전송
            files.append(('files', (filename, file.read(), 'image/png')))
    
    # 여러 파일을 한 번에 전송
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Failed to get prediction. Status code: {response.status_code}")

# 여러 프레임을 한 번에 전송
send_frames_batch("/home/insung/fall_detection/data/test/good/5", 50)
