Unsupervised dtCSM readme file
------------------------------

This Matlab code comes with two examples to run dtCSM from

1)test_artificalData.m

Uses three multivariate gaussian distributions (dimensionality = 50) and maps it to a 2D space.
You can play with different parameter options. 5 iterations is enough to see a good embedding of the data.

2) test_mnistData.m
Uses a subset of MNIST handwritten digit datasets (6000 digits) and maps it to a 2d space.
If you don't want to wait, you can open the jpg file to see the final result (200 iterations are used)

Despite of the separations of the different labels is not perfect, from an unsupervised point of view it makes sense since only distances on the binary raw representation were used. That's why 4 and 9 are intermixed, and also 5 is intermixed with 6 and 9. Unambiguous numbers like 0 and 1 are well differentiated


----------------------------------------------

Axel J. Soto, Ryan Kiros, Vlado Keselj, Evangelos Milios "Exploratory Visual Analysis and Interactive Pattern Extraction from Semi-Structured Data" Under review.

