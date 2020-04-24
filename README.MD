Phase 1 : Sans Cuda (en c).
Phase 2 : Cuda "simple"
Phase 3 : Cuda "shared"
Phase 4 : Cuda simple & streams
Phase 5 : Cuda "shared" & streams

10 convolutions - 9 simples et 1 composée

2 images : in.jpg (1920x1200) et in2.jpg (530x230)

3 configurations à tester :

	- de base :   
		  dim3 t( 32, 32 );
 		  dim3 bu( 3 * (( cols - 1) / (t.x + 1) , ( rows - 1 ) / (t.y) + 1 );


	- variante 1 :   
		  dim3 t( 1, 1 );
		  dim3 be( cols ), ( rows ); // variantes pour appels shared

	- variante 2 :   
		  dim3 t( 16, 16 );
		  dim3 be( 3 * (( 2 * cols - 1) / (t.x + 1) ), (( 2 * rows  - 1) / (t.y + 1) )); // variantes pour appels shared