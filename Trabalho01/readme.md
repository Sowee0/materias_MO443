Sintaxe para o arquivo trabalho01.py

python trabalho01.py -c/-g arquivo

Flags:
	-c/-g: 		Define se o processamento será a filtragem convolutiva em imagens de escala de cinza (grayscale, g) 
				ou a readequação de imagens coloridas (color, c).
				Uma imagem em cores pode ser usada para o processo de escala de cinza, mas será convertida.

 	arquivo:	Arquivo a ser processado

Resultado:
	-c: Duas imagens com nome arquivo_sepia e arquivo_escalaCinza
	
	-g: Dez imagens referentes à convolução com os filtros H1-H9 e H12 que é associação dos filtros H1 e H2
