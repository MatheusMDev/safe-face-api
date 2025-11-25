# SafeFaceAPI

API de reconhecimento facial desenvolvida em **Python** com **FastAPI** e **TensorFlow**.  
Este projeto serve como base inicial para autenticação e processamento de imagens com inteligência artificial(redes neurais).

---

## Tecnologias utilizadas
- **FastAPI** → criação da API e documentação automática (Swagger)
- **Uvicorn** → servidor ASGI para rodar a API
- **TensorFlow** → rede neural para reconhecimento facial
- **Pillow** → manipulação de imagens
- **Python-Multipart** → upload de arquivos via API

---

## 1 - Instalação
Clone este repositório:
git clone https://github.com/MatheusMDev/safe-face-api.git


### 2 - Crie o Ambiente virtual Python
python -m venv venv

#### 3 - Rode o seguinte comando no terminal 
venv\Scripts\activate

### 4 - Instale as Bibliotecas
pip install -r requirements.txt

### 5 - Rode o programa
cd .\api\
fastapi dev main.py
python -m uvicorn main:app --reload
