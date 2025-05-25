<p align="center">
  <svg width="35" height="35" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
    <path d="M13.873 3.805C21.21 9.332 29.103 20.537 32 26.55v15.882c0-.338-.13.044-.41.867-1.512 4.456-7.418 21.847-20.923 7.944-7.111-7.32-3.819-14.64 9.125-16.85-7.405 1.264-15.73-.825-18.014-9.015C1.12 23.022 0 8.51 0 6.55 0-3.268 8.579-.182 13.873 3.805ZM50.127 3.805C42.79 9.332 34.897 20.537 32 26.55v15.882c0-.338.13.044.41.867 1.512 4.456 7.418 21.847 20.923 7.944 7.111-7.32 3.819-14.64-9.125-16.85 7.405 1.264 15.73-.825 18.014-9.015C62.88 23.022 64 8.51 64 6.55c0-9.818-8.578-6.732-13.873-2.745Z" fill="#0085ff"/>
  </svg>
</p>

<h1 align="center">BskyMood</h1>

<p align="center">
  <strong>Coleta e Análise de Sentimentos em Tempo Real no Bluesky</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-Active-brightgreen.svg?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT">
</p>

---

## 📝 Sobre o Projeto

**BskyMood** é uma aplicação web desenvolvida em Python com Streamlit, projetada para interagir com a rede social Bluesky. A ferramenta permite coletar publicações (<em>skeets</em>) em tempo real através da API Firehose, realizar uma análise de sentimentos multilíngue (inglês, português e espanhol) sobre o conteúdo textual dessas publicações e, em seguida, apresentar os resultados de forma interativa. Os dados coletados, enriquecidos com a classificação de sentimento (positivo, negativo ou neutro), podem ser visualizados e descarregados em formato JSON.

Este projeto visa oferecer uma maneira prática de observar e analisar as tendências de sentimento e as conversas que ocorrem na plataforma Bluesky.

---

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como parte dos requisitos da disciplina **Tópicos Avançados em Ciências de Dados**, ministrada pelo **Prof. Alexandre Vaz**, no curso de Ciência de Dados e Inteligência Artificial do **Centro Universitário IESB**, em Brasília - DF.

---

## ✨ Funcionalidades Principais

* **Coleta em Tempo Real**: Conecta-se ao Firehose do Bluesky para capturar publicações assim que são criadas.
* **Filtragem de Idioma**: Foca em publicações nos idiomas inglês, português e espanhol.
* **Pré-processamento de Texto**: Limpa o texto das publicações removendo menções, URLs e outros ruídos antes da análise.
* **Análise de Sentimentos Multilíngue**: Utiliza o modelo `lxyuan/distilbert-base-multilingual-cased-sentiments-student` da Hugging Face para classificar o sentimento das publicações como positivo, negativo ou neutro.
* **Interface Interativa com Streamlit**:
    * Permite ao usuário definir a duração da coleta.
    * Exibe o status da coleta e da análise de sentimentos.
    * Apresenta os dados coletados e os resultados da análise em tabelas e métricas resumidas.
    * Oferece botões para iniciar/parar a coleta, reiniciar o processo e analisar sentimentos.
* **Download de Dados**: Permite fazer o download os dados coletados (incluindo a análise de sentimento) em formato JSON.
* **Threading e Multiprocessing**: Utiliza threads para a coleta de dados em segundo plano, garantindo que a interface do usuário permaneça responsiva, e `multiprocessing.Queue` para comunicação segura entre a thread de coleta e o processo principal.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.10+**
* **Streamlit**: Para a criação da interface web interativa.
* **AT Protocol SDK (`atproto`)**: Para interagir com a API Firehose do Bluesky.
* **Hugging Face Transformers (`transformers`)**: Para carregar e utilizar o modelo de análise de sentimentos.
* **Langdetect**: Para a detecção do idioma das publicações.
* **Regex**: Para o pré-processamento e limpeza de texto.
* **Threading & Multiprocessing**: Para operações concorrentes e responsividade da UI.

---

## 🚀 Como Acessar

Acesse o app através do Streamlit Community Cloud: [**BskyMood**](https://bskymood.streamlit.app).

---

## 📊 Exemplo de Uso

1.  Ao abrir o app, você verá a interface inicial.
2.  Na barra lateral, defina a **Duração da Coleta** desejada em segundos.
3.  Clique em **Iniciar Coleta**. O app começará a buscar publicações do Bluesky.
4.  Um botão **Parar Coleta** aparecerá, permitindo interromper o processo a qualquer momento.
5.  Após a coleta (ou interrupção), os dados brutos serão exibidos.
6.  Clique em **Analisar Sentimentos** para processar as publicações coletadas.
7.  Os resultados da análise, incluindo a classificação de sentimento para cada post e métricas agregadas, serão exibidos.
8.  Utilize o botão **Baixar Dados** para salvar os resultados em um arquivo JSON.
9.  Clique em **Reiniciar Coleta** para limpar todos os dados e começar novamente.


---

## 🤝 Contribuições

Contribuições são bem-vindas! Se você tiver sugestões para melhorar o BskyMood, sinta-se à vontade para abrir uma *issue* ou enviar um *pull request*.

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o ficheiro `LICENSE` para mais detalhes.

---

<p align="center">
  Feito com ❤️ e Python
</p>
