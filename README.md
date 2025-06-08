<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Bluesky_Logo.svg/869px-Bluesky_Logo.svg.png" alt="Bluesky Logo" width="32"/>
</p>

<h1 align="center">BskyMood</h1>

<p align="center">
  <strong>Coleta, Análise de Sentimentos e Modelagem de Tópicos em Tempo Real no Bluesky</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-Active-brightgreen.svg?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-GPL--3.0-yellow.svg?style=for-the-badge" alt="License: GPL-3.0">
</p>

<p align="center">
  <img src="https://github.com/jobsrobson/Streamlit-Bsky/blob/main/screenshot_gradia.png?raw=true" alt="Screenshot do BskyMood" width="800"/>
</p>


## 📝 Sobre o Projeto

**BskyMood** é uma aplicação web desenvolvida em Python com Streamlit, projetada para interagir com a rede social Bluesky. A ferramenta permite coletar publicações (*skeets*) em tempo real, realizar uma **análise de sentimentos** multilíngue (inglês, português e espanhol) e, em seguida, aplicar técnicas de **modelagem de tópicos com BERTopic** para descobrir os principais temas de discussão.

O aplicativo agrega os sentimentos por tópico, oferecendo insights sobre a percepção geral de cada tema, e apresenta os resultados em uma interface interativa. Os dados coletados, enriquecidos com as análises, podem ser visualizados e descarregados em formato JSON.

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como parte dos requisitos da disciplina **Tópicos Avançados em Ciências de Dados**, ministrada pelo **Prof. Alexandre Vaz**, no curso de Ciência de Dados e Inteligência Artificial do **Centro Universitário IESB**, em Brasília - DF.

## ✨ Funcionalidades Principais

* **Coleta em Tempo Real**: Conecta-se ao Firehose do Bluesky para capturar publicações assim que são criadas.
* **Filtragem de Idioma**: Foca em publicações nos idiomas inglês, português e espanhol.
* **Análise de Sentimentos Multilíngue**: Utiliza o modelo `lxyuan/distilbert-base-multilingual-cased-sentiments-student` da Hugging Face para classificar o sentimento de cada post.
* **Modelagem de Tópicos com BERTopic**: Identifica automaticamente os temas latentes nas publicações coletadas, agrupando conversas por similaridade semântica.
* **Análise de Sentimento Agregada**: Após a identificação dos tópicos, calcula e exibe a distribuição de sentimentos (positivo, negativo, neutro) para cada um deles.
* **Interface Interativa com Streamlit**:
    * Permite ao usuário definir a duração da coleta.
    * Apresenta os dados coletados e os resultados das análises em tabelas e métricas.
    * Controla o fluxo de análise com botões para iniciar a coleta, analisar sentimentos e, em seguida, analisar tópicos.
    * Exibe visualizações interativas dos tópicos, como o Mapa de Distância Entre Tópicos e o Gráfico de Palavras por Tópico.
* **Download de Dados**: Permite baixar todos os dados coletados e enriquecidos (sentimento e ID do tópico) em formato JSON.
* **Execução Concorrente**: Utiliza `threading` para a coleta de dados em segundo plano, garantindo que a interface do usuário permaneça sempre responsiva.

## 🛠️ Tecnologias Utilizadas

* **Python 3.10+**
* **Streamlit**: Para a criação da interface web interativa.
* **BERTopic**: Para a modelagem de tópicos.
* **Hugging Face Transformers**: Para carregar e utilizar o modelo de análise de sentimentos.
* **AT Protocol SDK (`atproto`)**: Para interagir com a API Firehose do Bluesky.
* **Scikit-learn**: Dependência para `BERTopic` e vetorização de texto.
* **NLTK**: Para processamento de linguagem natural (stopwords).
* **Langdetect**: Para a detecção do idioma das publicações.
* **Pandas**, **Regex**, **Threading**.

## 🚀 Como Executar o Projeto

Devido à alta demanda de recursos (RAM e CPU) dos modelos de Machine Learning, a execução na nuvem gratuita do Streamlit (Community Cloud) não é estável. A execução é recomendada através do **Google Colab**, que oferece um ambiente mais robusto.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Z01zVHUmpupHSprcJwtO1Sdh9zN7tKO3?usp=sharing)

**Clique no botão acima para abrir o notebook no Google Colab** e siga as instruções contidas nele para instalar as dependências e executar o aplicativo.

Se você quiser usar o Streamlit Community Cloud, basta acessá-lo abaixo. Lembre-se que a execução do BskyMood não é garantida e pode demorar ou falhar neste ambiente.

[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://bskymood.streamlit.app/)

## 📊 Exemplo de Uso

1.  Abra o notebook no Google Colab e execute as células de instalação e configuração.
2.  Na célula final, uma URL pública será gerada. Abra-a em seu navegador.
3.  Na interface do BskyMood, defina a **Duração da Coleta** desejada na barra lateral.
4.  Clique em **Iniciar Coleta**.
5.  Após a coleta, os dados brutos serão exibidos. Clique em **Analisar Sentimentos**.
6.  Com os sentimentos analisados, o botão **Analisar Tópicos** será habilitado. Clique nele.
7.  Explore os resultados! Navegue pela tabela de tópicos, os gráficos interativos e a tabela de dados detalhados, que agora inclui a classificação de sentimento e o ID do tópico para cada post.
8.  Utilize o botão **Baixar Dados** para salvar um arquivo JSON completo com os resultados.
9.  Clique em **Reiniciar Coleta** para limpar a memória e começar uma nova análise.

## 🤝 Contribuições

Contribuições são bem-vindas! Se você tiver sugestões para melhorar o BskyMood, sinta-à-vontade para abrir uma *issue* ou enviar um *pull request*.

## 📄 Licença

Este projeto está licenciado sob a Licença GPL-3.0. Veja o arquivo `LICENSE` para mais detalhes.

<p align="center">
  Feito com ❤️ e Python
</p>
