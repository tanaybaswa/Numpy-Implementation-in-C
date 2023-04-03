# numc

Here's what I did in project 4:
-
In this project I implemented a version of numpy. First, I wrote naive solutions for various matrix functions in C. This included an allocate function for new matrices, an allocate_ref function for matrices that were actually a slice of another matrix, a deallocate function to delete matrices, and arithmetic functions such as add, subtract, negate, absolute value, multiply, fill, and power. Most of these were done in a linear, piece-wise manner, and obviously the speedup was very low. The ultimate functionality of this project was to enable the user to call an executable file numc, and then write in python methods to modify and do math on matrices. 

These are some examples:

>>>import numc as nc #imports the module

>>>mat = nc.Matrix(3,3)   #matrix of 0's assigned to mat

>>>mat1 = nc.Matrix(3,3,1) #matrix of 1's assigned to mat1

>>>mat2 = nc.Matrix([1,2,3], [4,5,6]) #matrix as specified assigned to mat2

>>>mat3 = nc.Matrix(1,1,[2])  #matrix with 1 row and 1 column with value 2 assigned to mat3

>>>slice = mat2[1]        #slice gets row 1 of mat2
>>>slice
[4,5,6]

>>> x = mat + mat1  #element wise addition, now x holds a matrix of 1's
>>> a = mat - mat1  #subtraction, a holds a 3x3 matrix of -1's
>>> b = pow(mat1, 10) #matrix exponentiation, b = mat1 to the power of 10
>>> c = mat1 * mat   #matrix multiplication

>>>del a #delete matrix a
>>>del mat2 #delete matrix but keep its data as it has a slice
>>>del slice #delete matrix reference and its data

Now, after implementing the matrix functions in C in matrix.c, which worked on matrix objects, we now move to building the Python-C interface, which involved many specific Python-C interface structs and methods. I filled in numc.c according to the online API and the spec. I also wrote a setup file in python to include the correct flags and allow the program to compile correctly.In this, I had to wrap all the matrix functions I wrote in matrix.c so they would be called correctly in the Python interface. I also checked for errors in type or value of the passed in matrices.

Now it was time to speed everything up. I started by removing all my excessive function calls to get and set, as I could just directly use index arithmetic to get the right element. Then I performed loop unrolling for each of my simple fucntions which were add, sub, neg, abs, and fill. I started with an unrolling of 4 indices per loop. After this, I worked on vectorizing while keeping the loop unrolled, which resulted in incrementing by 16s, as the SIMD instructions worked with 4 doubles at a time. I used the AVX instructions relating to add, sub, load, store, and set to achieve this for all of the simple functions. Then I worked on matmul, and chose to use a transpose approach, which would implictly use blocking for cache hits. I allocated a new matrix, filled its data with the transpose of mat2 (sped up using SIMD and unrolling and index math). Then I simply computed more index math to find the right location to start the element-wise multiplication to fill the result matrix, used SIMD and unrolling with 32 sized jumps, and handled the tail cases appropriately. For Pow, I had a very basic recursive function, that built an identity when the pow == 0, simply memcpy'ed mat->data into result->data when pow == 1, and used repeated squaring when for all higher powers. These steps were enough to help me achieve the necessary speed up. 


