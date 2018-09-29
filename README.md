Fundamentação Teórica
==========

Introdução
----------

A separação de regiões correspondentes à pele humana em imagens digitais possui fundamental importância para a solução de problemas relacionados à visão computacional, como detecção de face, detecção de gestos e classificação de conteúdo. Tais aplicações empregam algoritmos de segmentação com o objetivo de delimitar áreas de interesse nas imagens, de modo a reduzir o escopo de informações a serem processadas e obter ganhos de desempenho.

Diversas são as técnicas envolvidas na tarefa de segmentação de imagens, como k-means e outras formas de reconhecimento de padrões. Porém, devido às suas propriedades físicas e forma peculiar de interagir com a luz, a pele humana possui características visuais que a diferenciam de elementos inorgânicos, o que traz vantagem a abordagens baseadas na cor. De fato, o espectro de tonalidades da pele humana é relativamente limitado, mesmo levando em consideração variações étnicas.

O problema resume-se, então, a determinar se um dado conjunto de pixels pertence ou não a uma região de pele humana com base em sua cor. Nesse contexto, a escolha do espaço de cores mais adequado aos matizes da pele humana, bem como o emprego de um modelo estatístico que permita analisar um conjunto previamente classificado de imagens pode ser uma solução eficaz. Tais escolhas estão intrinsecamente relacionadas à sensibilidade dos algoritmos de segmentação a problemas como condições de iluminação desfavoráveis e oclusão.

Este trabalho propõe o uso de um classificador Naive Bayes para esta tarefa, empregando duas classes de dados em seu modelo estatístico: pele e não-pele. Busca-se assim, obter um algoritmo de segmentação por cor dotado de bons níveis de precisão e velocidade suficiente para ser aplicado em filmagens em tempo real em um computador pessoal. A seção materiais e métodos descreve os conceitos matemáticos empregados; posteriormente, a seção análise mostra os resultados obtidos e comparações de desempenho em diferentes espaços de cor; por fim, a seção Conclusão e Perspectivas Futuras interpreta os resultados e sintetiza as linhas de pesquisa futuras para este trabalho.

Materiais e métodos
-------------

O teorema de Bayes, nomeado em homenagem a seu idealizador, Thomas Bayes (1701-1761), estabelece uma relação matemática para as probabilidades de eventos condicionados a evidências prévias. A probabilidade de um evento A, dado que houve a observação de uma evidência B é descrita por:

![](https://latex.codecogs.com/gif.latex?P%28A%7CB%29%3D%5Cfrac%7BP%28B%7CA%29%5Ctimes%7BP%28A%29%7D%7D%7BP%28B%29%7D).

Este princípio possui diversas aplicações no campo da inferência estatística, em problemas que demandam a dedução de informações a partir da análise de um conjunto de amostras. Filtros de spam, por exemplo, analisam o texto de diversos emails classificados pelos usuários como spam e não-spam e determinam a classe na qual se enquadra uma nova amostra com base na equação acima.

Classificadores Naive Bayes consideram que as características analisadas são independentes entre si. Isto é, para um conjunto ![](https://latex.codecogs.com/gif.latex?A%3D%7BA_1%2CA_2%2C%5Ccdots%2CA_n%7D) de características condicionadas a uma evidência ![](https://latex.codecogs.com/gif.latex?B), tem-se o produtório:

![](https://latex.codecogs.com/gif.latex?P%28A_1%2CA_2%2C%5Ccdots%2CA_n%7CB%29%3D%5Cprod_%7Bi%3D1%7D%5E%7Bn%7D%20P%28A_i%7CB%29.)

Tal pressuposto, embora possa comprometer a coerência entre o modelo e a relação que de fato ocorre entre os dados, é capaz de classificar as amostras com níveis de erro próximos aos de métodos mais robustos.

Esta modelagem matemática pode ser aplicada para a segmentação de imagens como uma técnica de classificação de pixels em duas classes: pele e não-pele. Para imagens representadas em espaços de cor de 3 canais, por exemplo, considera-se que o evento A representa o fato de um pixel pertencer a uma região de pele. B corresponde ao valor numérico que um pixel assume em determinado canal. Logo, primeiramente é necessário obter ![](https://latex.codecogs.com/gif.latex?P%28A%29), ![](https://latex.codecogs.com/gif.latex?P%28B%7CA%29) e ![](https://latex.codecogs.com/gif.latex?P%28B%29), conhecidos como probabilidades *a priori*. De forma intuitiva, tem-se, pois:

-   ![](https://latex.codecogs.com/gif.latex?P%28B%7CA%29) = probabilidade de dada cor ser pele;

-   ![](https://latex.codecogs.com/gif.latex?P%28A%29) = probabilidade de encontrar dada cor;

-   ![](https://latex.codecogs.com/gif.latex?P%28B%29) = probabilidade de encontrar pele.

![Exemplo de imagem do conjunto classificado manualmente.](https://s3-sa-east-1.amazonaws.com/abnersn/github/bayesian-segmentator/demo.jpg)

A partir de um conjunto de imagens como a figura acima, é possível calcular as probabilidades *a priori* necessárias para a aplicação do teorema de Bayes. Após a análise de todas as imagens em um conjunto pré classificado com a ajuda de um editor de imagens, os valores obtidos das probabilidades para cada canal são armazenados em uma tabela de referência.

Ao receber uma imagem inédita, o algoritmo busca, para cada pixel, uma probabilidade correspondente na tabela, de acordo com os valores de seus canais, isto é, sua cor. Em seguida os valores obtidos são multiplicados, conforme a Equação \[eq:2\], para obter a classificação final. Dessa forma, a imagem se torna uma matriz de probabilidades inferidas, com valores entre 0 e 1. Determina-se, então, um limiar $\lambda$, de modo que:

$$C_A(p_{i}) = \left\{
        \begin{array}{ll}
            0 & \quad p_i \leq \lambda \\
            1 & \quad p_i > \lambda
        \end{array}
    \right..$$

Onde $p_i$ é a probabilidade inferida e $C_A$ corresponde à classe atribuída ao pixel, isto é, 0 para não-pele e 1 para pele.

Para avaliar a capacidade de acerto do algoritmo, utiliza-se a tabela de referência para classificar as imagens usadas no treino. Dessa forma, é possível comparar as probabilidades inferidas com a classificação realizada manualmente. Considerando $\epsilon$ o erro quadrático médio do algoritmo aplicado em uma imagem composta por $n$ pixels, obtém-se: $$\epsilon=\sum_{i=1}^{n}{\frac{(p_i - C_i)^2}{n}},$$ onde $C_i$ representa o valor atribuído ao $i$-ésimo pixel na classificação manual.

Resultados
==========

A Figura \[fig:result\] mostra o resultado obtido pelo classificador Naive-Bayes para diferentes espaços de cor, nomeadamente, HSV, YCrCb e RGB, bem como as respectivas taxas de erro quadrático médio. As imagens originais estão disponíveis no banco de imagens Wikimedia Commons e foram classificadas manualmente com o auxílio do editor de imagens GIMP. Neste trabalho, para a obtenção das probabilidades *a priori* foram empregadas 13 imagens, tomando a diversidade étnica como critério para sua escolha. As taxas de acerto obtidas variam, pois, de $88,4\%$ para o espaço RGB a $90,6\%$ no espaço YCrCb.

![Comparativo de desempenho e taxas de erro em diferentes espaços de cor.[]{data-label="fig:result"}](https://s3-sa-east-1.amazonaws.com/abnersn/github/bayesian-segmentator/result.png)