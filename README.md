# Workshop ML e GenAI - UNESP

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

Bem-vindo ao repositório oficial do **Workshop de Machine Learning e Inteligência Artificial Generativa (GenAI)** realizado na Unesp. Este projeto contém todo o código-fonte, dados e notebooks interativos utilizados durante a apresentação.

---

## 🎯 Visão Geral

O objetivo deste projeto é fornecer uma demonstração prática e educacional de algoritmos clássicos de Machine Learning e integrações modernas com Modelos Fundacionais (Generative AI), incluindo modelos da familia **Gemini (Vertex AI)** e **Stable Diffusion**.

O repositório foi construído seguindo princípios de **Engenharia de Software (SOLID e Clean Code)**, com modularização e fácil reprodutibilidade.

---

## 🏗️ Estrutura do Projeto

Abaixo está a representação da arquitetura de pastas do projeto:

```text
Workshop-ML-e-GenAI-Unesp/
├── data/
│   └── raw/             # Bases de dados em CSV e imagens em ZIP
├── docs/                # Slides e PDFs da apresentação
├── notebooks/           # Jupyter Notebooks interativos da demonstração
├── src/                 # Código modular em Python
│   ├── data/            # Scripts para carregamento e pré-processamento de dados
│   ├── models/          # Wrappers para ML clássico e IA Generativa
│   └── utils/           # Funções de visualização e manipulação de imagens
├── tests/               # Testes unitários (pytest)
├── README.md            # Este arquivo
├── CONTRIBUTING.md      # Guia para contribuição no projeto
└── requirements.txt     # Dependências do projeto
```

---

## 🚀 Pré-requisitos e Instalação

### Pré-requisitos
- Python 3.9 ou superior
- Ambiente Virtual (recomendado)
- Para rodar os modelos GenAI, você precisará de acesso ao Google Cloud Vertex AI ou tokens das respectivas APIs (Cohere, Hugging Face).

### Passo a Passo de Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/Workshop-ML-e-GenAI-Unesp.git
   cd Workshop-ML-e-GenAI-Unesp
   ```

2. **Crie um ambiente virtual (Opcional, porém recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Exemplos de Uso

A lógica de negócio foi abstraída na pasta `src/` para manter a demonstração limpa e concisa.

### Utilizando Machine Learning Clássico
```python
from src.data.data_loader import load_csv_data, preprocess_regression_data
from src.models.classical import SimpleLinearRegression

# Carrega os dados
df = load_csv_data("data/raw/boston.csv", sep=";", decimal=",")
X, y = preprocess_regression_data(df, target_col="MV")

# Treina o modelo
modelo = SimpleLinearRegression()
modelo.fit(X, y)
previsoes = modelo.predict(X)

# Avalia o modelo
modelo.evaluate(y, previsoes)
```

### Utilizando GenAI (Vertex AI / Gemini)
```python
from src.models.genai import GeminiWrapper
from src.utils.visual import load_image_from_url

# Inicializa o wrapper do Vertex AI
gemini = GeminiWrapper()

# Carrega imagem
img = load_image_from_url("https://exemplo.com/imagem.jpg")

# Prepara os inputs e gera o resultado
prompt = "Descreva o que há na imagem:"
for texto in gemini.generate_content_stream([prompt, img]):
    print(texto, end="")
```

---

## ✅ Execução de Testes

Para garantir o funcionamento da biblioteca utilitária do repositório, execute a suíte de testes com `pytest`:

```bash
pytest tests/
```

---

## 🤝 Como Contribuir

Fique à vontade para propor melhorias! Por favor, consulte nosso [Guia de Contribuição](CONTRIBUTING.md) para maiores detalhes sobre os padrões adotados neste projeto.

---

## 📝 Licença

Este projeto está sob a licença MIT. Sinta-se livre para usá-lo para fins acadêmicos e comerciais.
