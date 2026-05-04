# Guia de Contribuição

Obrigado por se interessar em contribuir para o repositório **Workshop ML e GenAI Unesp**! Este guia descreve o processo para enviar melhorias, correções de bugs e novos exemplos.

## Como Contribuir

1. Faça um **Fork** deste repositório.
2. Clone o repositório para sua máquina local:
   ```bash
   git clone https://github.com/SEU_USUARIO/Workshop-ML-e-GenAI-Unesp.git
   ```
3. Crie uma nova branch para a sua funcionalidade ou correção:
   ```bash
   git checkout -b feature/minha-nova-funcionalidade
   ```
4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
5. Faça as modificações necessárias. **Lembre-se de adicionar testes para novas funcionalidades**.
6. Execute os testes para garantir que tudo continua funcionando:
   ```bash
   pytest
   ```
7. Faça o *commit* das suas alterações com mensagens claras e descritivas:
   ```bash
   git commit -m "feat: adiciona nova função de visualização de mapas de calor"
   ```
8. Envie para o GitHub:
   ```bash
   git push origin feature/minha-nova-funcionalidade
   ```
9. Abra um **Pull Request** detalhando as alterações realizadas.

## Padrões de Código
- Utilizamos **Clean Code** e **SOLID**.
- Adicione **Type Hinting** a todas as funções.
- Mantenha funções focadas em fazer uma única coisa bem feita.
- Documente suas funções utilizando o padrão de docstrings em Python.

## Reportando Bugs
Se você encontrar algum bug ou problema, por favor abra uma **Issue** no repositório descrevendo o cenário, os passos para reproduzir o problema e o comportamento esperado.
