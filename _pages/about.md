---
title: 캡스톤 디자인 프로젝트
author: Cho Ye Eun
date: 2024-01-18
category: Jekyll
layout: post
---

**졸업 프로젝트 주제 요약 : 사용자의 얼굴을 기반으로 생성된 아바타를 곁들여 몰입도를 높인 역사 AR 어플리케이션**

현재 졸업프로젝트에서 제가 맡고있는 역할은 백엔드입니다.

**1\. 촬영된 얼굴 사진을 딥러닝 모델로 전송. 모델은 사진을 보고 얼굴의 각 부분에서 landmark를 추출함  
****2\. 딥러닝 모델이 출력한 얼굴 landmark 결과값을 다시 유니티로 전송. 해당 결과값은 아바타를 생성하는 것에 이용됨**

위 프로세스가 돌아가도록 하는 것이 제가 해야하는 일입니다.

그래서 이번 포스팅에서는 위 프로세스를 작동시키기 위한 저의 눈물나는 뭐시깽이를 정리해보고자 합니다.(>.ㅇ)

---

### **1\. AWS EC2 instance 생성하기**

이것과 관련해서 제가 이전에 정리해 둔 포스팅이 있습니다. 해당 포스팅에서는 AWS 가입부터 EC2 instance 생성, ssh을 이용한 instance 접근까지 모두 설명되어 있습니다.

[https://schoolhomeblog.tistory.com/4](https://schoolhomeblog.tistory.com/4)

 [\[졸업프로젝트/2023-02-02(목)\] AWS ec2 를 이용하여 서버 접속하기(MacOS)

\* 원래는 '\[졸업프로젝트/2023-02-02(목)\] 웹 AR 조사 및 AWS ec2 를 이용하여 서버 접속하기(MacOS)'으로, '웹 AR 조사'와 'AWS ec2 를 이용하여 서버 접속하기(MacOS)'가 하나의 포스팅으로 되어있었는데 내용

schoolhomeblog.tistory.com](https://schoolhomeblog.tistory.com/4)

\* 인스턴스는 생성 후 중지 시키고 다시 시작할 때 마다 IP가 변경되게 됩니다. 이 점 참고 부탁드립니다.

\* '종료'가 아닌 '중지'입니다. 인스턴스를 '종료'하실 경우, 해당 인스턴스를 다시 사용할 수 없게 됩니다. 종료 = 삭제 라고 생각하시면 됩니다.

\* 저의 경우 프리티어 요금제를 사용하고 있으며, 인스턴스를 시작하고 중지하는걸 반복하며 사용하고 있습니다.

### **2\. flask 어플리케이션 작성하기**

flask는 python 프레임 워크 중 하나입니다. 딥러닝 모델 배포를 위해 사용하려고 합니다.

python이 설치된 상태에서 터미널에 pip install flask 을 입력해 flask를 설치합니다.

\* python이 설치되어 있지 않은 경우 homebrew가 설치되어 있다는 전제 하에 터미널에서 brew install python3 명령어를 실행해서 설치할 수 있습니다.(참고 : [https://zarazio.tistory.com/11](https://zarazio.tistory.com/11))

에디터에서 파이썬으로 app.py를 작성해줄겁니다.

\* 에디터와 관련해서 저의 경우 xcode를 사용하였는데, app store에서 설치해도 되고 직접 application 파일을 다운 받아 설치해도 상관 없습니다만 후자가 좀 더 빠릅니다.(참고 : [https://cosmosproject.tistory.com/484](https://cosmosproject.tistory.com/484))

\* xcode 말고 vscode를 사용해 줄 거라면, python 확장 기능을 install 해도 괜찮습니다(참고 : [https://www.codingfactory.net/11337](https://www.codingfactory.net/11337))

app.py 코드의 기능은 전송된 이미지(바이트 형식으로 변환된 상태)를 읽어 모델(shape\_predictor\_81\_face\_landmarks.dat)에 돌린 후 모델이 추출한 landmark의 값을 json 형태로 반환하는 것입니다. 생각해둔 기능에 따라 app.py를 작성해줍니다.

```
from flask import Flask, request, jsonify
import dlib
import numpy as np
import cv2

app = Flask(__name__)
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

@app.route('/', methods=['POST','PUT'])
def predict():
    
    # 이미지를 받아와서 얼굴 랜드마크 예측
    img = request.data
    npimg = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if not rects:
        return jsonify({'error': 'no face detected'}), 400
    shape = predictor(gray, rects[0])

    # 얼굴 랜드마크 좌표를 리스트에 저장
    landmarks = []
    for i in range(81):
        landmarks.append([shape.part(i).x, shape.part(i).y])

    # 얼굴 랜드마크 좌표를 JSON 형식으로 반환
    return jsonify({'landmarks': landmarks})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

보통 flask 어플리케이션은 port를 5000으로 설정해주는 경우가 많다고 해서, port를 5000으로 설정해주었습니다.

### **3\. AWS EC2 instance 보안 설정하기**

1번에 따라 인스턴스를 생성해 주셨으면, 보안 설정을 수정해 통신에 사용할 포트를 미리 허용해두어야 합니다.

2번에서 5000번 포트로 설정했으니, 5000번 포트를 허용하는 규칙을 넣어보도록 하겠습니다.

먼저, 보안 설정을 할 수 있도록, 관련 페이지에 들어가보도록 하겠습니다.

[##_Image|kage@b0savC/btr9mRNqAlS/2e7sNkdrRzaKcjDr3hKMQK/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"보안규칙을 설정해주려는 인스턴스 ID 선택","filename":"스크린샷 2023-04-10 오후 4.08.15.png"}_##][##_Image|kage@cliltf/btr9mNjZXC7/8j8JwMxWcEgh7qolu7fCT0/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"아래로 드래그","filename":"edited_스크린샷 2023-04-10 오후 4.08.23.png"}_##][##_Image|kage@yMooz/btr8ZVDRwXZ/yL3J47Kd7ZNx4vtaYEilt0/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"[보안] 선택 후 옆으로 드래그","filename":"edited_스크린샷 2023-04-10 오후 4.08.36.png"}_##][##_Image|kage@83sHb/btr88rbjiJU/zwJKYyzztxvtqKaMtYb5u0/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"보안 그룹 선택","filename":"스크린샷 2023-04-10 오후 4.08.51.png"}_##]

인바운드 규칙을 먼저 수정해주겠습니다. 인바운드 규칙은 해당 인스턴스로 접근하는 것과 관련된 규칙입니다.

[##_Image|kage@bedjWi/btr891Kuegu/96u3m01MFEOP96uMROuPXK/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"[인바운드 규칙] 선택","filename":"스크린샷 2023-04-10 오후 4.09.04.png"}_##][##_Image|kage@B6flT/btr9nIoYgdg/CfFDReuHa7GxJlXw4tJKHk/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"[인바운드 규칙 편집] 선택","filename":"스크린샷 2023-04-10 오후 4.09.09.png"}_##]

\[규칙 추가\]를 선택해 5000포트의 접근을 허용하는 규칙을 추가해줍니다.

[##_Image|kage@Nscaq/btr9n6pKLGr/71611mkUbyd7JUTzG4U0kk/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"[규칙 추가] 선택","filename":"스크린샷 2023-04-10 오후 4.09.14.png"}_##][##_Image|kage@4ffJS/btr8914OCus/lkiDnICw2EhNwrJvC648Yk/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"[규칙 저장] 선택","filename":"스크린샷 2023-04-10 오후 4.09.40.png"}_##][##_Image|kage@cWrcri/btr9nJVH3gg/dtwDfnNKGRzR8qT0PLw691/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"규칙이 추가된 것을 확인 후 위로 드래그","filename":"스크린샷 2023-04-10 오후 4.10.04.png"}_##]

이어서 아웃바운드 규칙을 수정해주겠습니다. 아웃바운드 규칙은 해당 인스턴스에서 내보내는 것과 관련된 규칙입니다.

[##_Image|kage@bg86xO/btr9nJBqjX7/8P8fgROqOUkAXJMdT6LNnK/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"[아웃바운드 규칙] 선택","filename":"스크린샷 2023-04-10 오후 4.10.09.png"}_##][##_Image|kage@rh4ib/btr9nIJg2UU/ep2mD7aU7ApV9eYtk8saa0/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"[아웃바운드 규칙 편집] 선택","filename":"스크린샷 2023-04-10 오후 4.10.15.png"}_##][##_Image|kage@AWxXk/btr9mRGEIaZ/xVorTNs5py8QrQ9cw4y4W0/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"[규칙 추가] 선택 후 5000 포트로 나갈 수 있도록 규칙 추가","filename":"스크린샷 2023-04-10 오후 4.10.20.png"}_##][##_Image|kage@bFNjna/btr9nIibwM9/CAOA0VQPjyvC1R4YaSWDGK/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"[규칙 저장] 선택","filename":"스크린샷 2023-04-10 오후 4.10.29.png"}_##][##_Image|kage@vAQFj/btr8XxXEKJb/gTf1ybwa5tJWosuHd1ln21/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"규칙이 추가된 것을 확인","filename":"스크린샷 2023-04-10 오후 4.10.41.png"}_##]

### **4\. flask 어플리케이션을 넣어 docker image 빌드하기**

2번 에서 작성한 app.py는 여러 의존성 패키지와 라이브러리를 import 합니다.

여유가 된다면 AWS EC2 instance에 해당 패키지 및 라이브러리를 설치해주면 되겠지만, AWS 프리티어 요금제에서는 다소 용량이 큰 라이브러리(예를 들어 opencv같은)를 직접 설치해서 사용하기에 애로사항이 꽃피더라구요.

그래서, docker file을 작성해 app.py와 의존성 패키지 및 라이브러리를 묶어서 docker image를 빌드한 후 인스턴스에 pull해보기로 했습니다.

#### **4-1. docker 및 docker buildx 설치**

먼저, docker을 설치해줘야 하는데, [해당 링크](https://docs.docker.com/desktop/install/mac-install/)에 들어가서 본인에게 맞는 버전으로 docker를 설치해주면 됩니다.

그리고 [docker hub](https://hub.docker.com/)에 회원가입을 해줍니다. 나중에 docker image를 빌드한 후 push/pull해주기 위해 필요합니다.

그 다음으로, buildx 명령어를 사용해주기 위해 buildx를 설치해줍니다. buildx를 설치하는 이유는 m1 맥북을 별다른 설정 없이 사용하면 arm64 아키텍처로 docker image가 빌드되는데, aws ec2 instance의 아키텍처를 amb64로 설정해 둔 상황이라 아키텍처의 차이로 인한 오류가 발생하게 됩니다. 이를 피하기 위해서 아키텍처를 amd 64로 빌드해줄 필요가 있고, 이 때 buildx를 사용합니다.

다음 명령어를 터미널에 입력하면 설치됩니다.

\* Docker의 버전이 19.03 이상인지 확인이 필요합니다. 'docker version' 명령어를 입력해주면 확인 가능합니다.

```
docker --version
docker buildx install
```

이후 'docker buildx' 명령어를 입력했을때 다음과 같은 결과가 나타다면, 'buildx'를 사용할 수 있습니다.

[##_Image|kage@dD6be7/btr9pYmFDml/7Ezf7RUWgfk1WwkrUhKQv1/img.png|CDM|1.3|{"originWidth":1222,"originHeight":768,"style":"alignCenter","filename":"스크린샷 2023-04-11 오후 5.10.08.png"}_##]

#### **4-2. dockerfile 작성 및 docker image 빌드**

dockerfile과 requirements.txt를 작성해줍니다.

**dockerfile**

더보기

```
FROM --platform=linux/amd64 python:3.9.10-slim-buster
WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev && pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .




EXPOSE 5000

CMD ["python3", "app.py"]
```

**requirements.txt**

더보기

```
Flask==2.0.2
dlib==19.21.0
numpy==1.21.2
opencv-python-headless==4.5.4.58
```

\* 에디터를 이용해서 작성해줘도 되지만 저는 그냥 텍스트 입력기로 작성했습니다.

작성한 dockerfile과 requirements.txt는 app.py와 같은 디렉토리에 넣어주어야 합니다.

저의 경우 savata\_docker라는 폴더를 만들어서 한 곳에 모아주었습니다.

[##_Image|kage@Ewm16/btr9pbLgpQ4/zfGLp2AnSLKHQXtkWpJ6dk/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","filename":"edited_스크린샷 2023-04-10 오후 5.01.51.png"}_##]

터미널로 들어간 후 빌드를 위한 명령어들을 입력하기 전, docker 프로그램을 눌러서 docker를 켜줍니다.

[##_Image|kage@kWCKl/btr9ppdzYhf/CQfaPejOFEPQiSlAAj5gOK/img.png|CDM|1.3|{"originWidth":2880,"originHeight":52,"style":"alignCenter","caption":"고래 모양이 나와있다면 docker가 켜져있다는 의미","filename":"edited_스크린샷 2023-04-11 오후 5.21.50.png"}_##]

docker에 먼저 로그인 해줍니다.

아래 명령어를 입력한 후, 비밀번호를 입력하고 엔터를 눌러 로그인해주시면 됩니다.

```
docker login -u [docker ID]
```

[##_Image|kage@n0krh/btr9yxICHcd/3v3aSzB8Pc4a0cqUQTPRhK/img.png|CDM|1.3|{"originWidth":1870,"originHeight":254,"style":"alignCenter","filename":"스크린샷 2023-04-11 오후 5.26.08.png"}_##]

그 다음 아래의 명령어를 입력하여 docker image를 build 하면서 동시에 docker hub로 push 해줍니다.

```
cd [app.py, dockerfile, requirements.txt 파일이 있는 폴더 경로]
docker buildx build --platform linux/amd64 -t [docker hub ID]/[docker image 명] --push .
```

[##_Image|kage@6c2ue/btr9pcFxRGQ/kNu6S9lfDdYwWjGaUvLIN0/img.png|CDM|1.3|{"originWidth":2880,"originHeight":826,"style":"alignCenter","caption":"빌드 완료","filename":"스크린샷 2023-04-11 오후 5.33.03.png"}_##][##_Image|kage@k6Is2/btr9yxu6vaF/SPewvAN59V53FB7Iky7Y80/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"docker hub에 잘 push 된 것을 확인함","filename":"스크린샷 2023-04-11 오후 5.34.38.png"}_##]

### **5\. docker image를 AWS EC2 instance에 pull 한 후 실행하기**

먼저, ssh를 이용해 EC2 인스턴스에 접근합니다.

\*자세한 내용은 [해당 포스트](https://schoolhomeblog.tistory.com/4)를 참고해 주세요.

```
cd [key 파일 존재 폴더 경로]
chmod 400 [key 파일 이름].pem
ssh -i ./[key 파일 이름].pem ubuntu@[IP 주소]
```

중간에 'Are you sure you want to continue connecting(yes/no/\[fingerprint\])?'라는 문장이 나오면 'yes' 입력해주면 됩니다.

[##_Image|kage@cnfOmo/btr9wHLzgkC/KjQlR2toHvyNRTB1a8rYlk/img.png|CDM|1.3|{"originWidth":1580,"originHeight":1266,"style":"alignCenter","caption":"ssh 접속됨","filename":"스크린샷 2023-04-11 오후 5.43.26.png"}_##]

#### **5-1. EC2 instance에 docker 설치 및 group 설정해주기**

인스턴스에 docker를 설치해 주어야 docker image를 pull한 후 실행시킬 수 있습니다.

ssh로 인스턴스에 접근한 상태에서, 다음 명령어를 차례대로 한줄씩 입력해서 docker를 설치해줍니다.

```
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

이후, 'sudo docker --version' 명령어를 입력해 제대로 설치가 되었는지 확인합니다.

[##_Image|kage@cYf0gg/btr9BH4zDAo/Z4NyqzdYWdXBhWQso91fMk/img.png|CDM|1.3|{"originWidth":860,"originHeight":68,"style":"alignCenter","filename":"스크린샷 2023-04-11 오후 5.53.55.png"}_##]

그 다음, 'groups' 명령어를 입력하여, 'docker' 그룹이 존재하는지 확인합니다. 아마 인스턴스 생성 후 위의 절차를 따라 docker만 설치해준 상태라면, 'docker' 그룹이 없는 상태일 것입니다.

아래 명령어를 순서대로 한 줄 씩 입력해서 'docker' 그룹을 만들고, 사용자를 'docker' 그룹에 추가해 줍니다.

```
sudo groupadd docker
sudo usermod -aG docker $USER
```

변경 사항이 저장되려면 다시 로그인 해주어야 해서, 'exit' 명령어를 입력하여 인스턴스에 대한 ssh 연결을 끊었다가 다시 연결해줍니다.

이후 'groups' 를 입력하면 'docker' 그룹이 추가되어 있는 것을 확인할 수 있습니다.

[##_Image|kage@b9lwye/btr9yx9MWAW/s9NrKSwfBdyAaClDDf1lt0/img.png|CDM|1.3|{"originWidth":1138,"originHeight":68,"style":"alignCenter","filename":"스크린샷 2023-04-11 오후 6.03.42.png"}_##]

#### **5-2. docker image를 EC2 instance로 pull 하기**

아래 명령어를 순서대로 입력하여 docker image를 인스턴스로 pull한 후 실행합니다.

```
docker pull [docker ID]/[docker image 이름]
docker run -d -p 5000:5000 [docker ID]/[docker image 이름]
```

[##_Image|kage@TK60D/btr9wIjslYj/m3ATBDE6DgonghutdKDmIK/img.png|CDM|1.3|{"originWidth":1294,"originHeight":488,"style":"alignCenter","filename":"스크린샷 2023-04-11 오후 6.17.40.png"}_##]

딥러닝 모델과 통신을 할 수 있도록 해주는 플라스크 어플리케이션의 배포가 완료되었습니다!

### **6\. 유니티 C# script 작성하기**

유니티에서 새로운 프로젝트를 만들어 준 후, 하단의 \[asset\]에 우클릭한 후 \[create\] - \[C# Script\]를 눌러 새로운 스크립트를 만들어줍니다.

[##_Image|kage@Eu1Kj/btr9pYUFafr/gAw3W5Po0fp1NUEKzjU441/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","filename":"스크린샷 2023-04-11 오후 6.22.12.png"}_##]

C# script를 작성해줍니다.

해당 스크립트의 기능은 다음과 같습니다.

1\. 이미지를 바이트 형식으로 서버로 전송하면서 서버의 PUT 메서드를 호출  
2\. landmark를 json 형식으로 반환받아 콘솔창에 출력

**FaceLandmarksController.cs**

더보기

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json.Linq;

public class FaceLandmarksController : MonoBehaviour
{
    private string url = "http://[IP 주소 입력]:5000";

    // Start is called before the first frame update
    void Start()
    {
        // Load test image
        Texture2D tex = Resources.Load<Texture2D>("iu_face");

        // Convert Texture2D to byte array
        byte[] bytes = tex.EncodeToPNG();

        // Send put request to server
        StartCoroutine(Put(bytes));
    }

    private bool IsValidImage(byte[] imageData)
    {
        // Check if image data is null or empty
        if (imageData == null || imageData.Length == 0)
        {
            Debug.LogError("Invalid image data: null or empty");
            return false;
        }

        // Check if image data can be decoded to a texture
        Texture2D tex = new Texture2D(2, 2);
        if (!tex.LoadImage(imageData))
        {
            Debug.LogError("Invalid image data: cannot be decoded to a texture");
            return false;
        }

        return true;
    }

    IEnumerator Put(byte[] imageBytes)
    {
        if (!IsValidImage(imageBytes))
        {
            yield break;
        }

        // Create UnityWebRequest object and set method to PUT
        UnityWebRequest request = UnityWebRequest.Put(url, imageBytes);
        request.method = "PUT";

        request.SetRequestHeader("Content-Type", "application/octet-stream");
        request.SetRequestHeader("Accept", "application/json");

        // Send the request and wait for the response
        yield return request.SendWebRequest();

        // Check for errors
        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError(request.error);
            yield break;
        }

        // Parse JSON response
        JObject response = JObject.Parse(request.downloadHandler.text);
        JArray landmarksArray = (JArray)response["landmarks"];

        // Convert JArray to 2D float array
        float[,] landmarks = new float[landmarksArray.Count, 2];
        for (int i = 0; i < landmarksArray.Count; i++)
        {
            landmarks[i, 0] = (float)landmarksArray[i][0];
            landmarks[i, 1] = (float)landmarksArray[i][1];
        }

        // Print landmarks to console
        for (int i = 0; i < landmarks.GetLength(0); i++)
        {
            Debug.Log("Landmark " + i + ": (" + landmarks[i, 0] + ", " + landmarks[i, 1] + ")");
        }
    }
}
```

해당 C# script를 실행시키기 위해 여러 작업이 필요합니다.

먼저, hierarchy 창에 우클릭 해서 \[Create Empty\]를 선택해준 후, 해당 오브젝트의 Inspector 창에서 \[Add Component\]를 선택해 방금 작성한 C# script를 추가해줍니다.

[##_Image|kage@b4jmca/btr9pZMQuOZ/5gZkD6G1s80ksFXeWJ8lZk/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","filename":"스크린샷 2023-04-11 오후 6.30.04.png"}_##][##_Image|kage@cZwAmv/btr9AYZUauE/K4K3FyvxvgNtb2kAf3g0U0/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","filename":"스크린샷 2023-04-11 오후 6.30.36.png"}_##][##_Image|kage@dzPzrM/btr9rgt5WCo/9QLKCueksCuT4dLr0QLIO1/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"작성한 C# script 추가","filename":"스크린샷 2023-04-11 오후 6.31.12.png"}_##]

그 다음, 하단의 \[asset\]에 우클릭한 후 \[create\] - \[Folder\]를 눌러 'Resources' 폴더를 만들어 줍니다.

[##_Image|kage@0lTCf/btr9zvqBlSJ/FDGSOZkqBBbkrbKzgNbNAk/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","filename":"스크린샷 2023-04-11 오후 6.31.48.png"}_##]

모든 일이 다 되었습니다. 이제 \[play\] 버튼을 눌러줍니다.

[##_Image|kage@RXhxM/btr9wIKBvdu/v2rGYLTGTK8zhEYKxULJsk/img.png|CDM|1.3|{"originWidth":2880,"originHeight":1800,"style":"alignCenter","caption":"성공!","filename":"KakaoTalk_Photo_2023-04-11-18-41-02.png"}_##]

야호! 성공!!