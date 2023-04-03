#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows < 1 || cols < 1){
    	return -1;
    }

    (*mat) = (struct matrix*) malloc(sizeof(struct matrix));
    if ((*mat) == NULL){
    	return -1;
    }

    (*mat)->data = (double *) calloc(rows * cols, sizeof(double));

    if ((*mat)->data == NULL){
    	return -1;
    }

    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;
    (*mat)->rows = rows;
    (*mat)->cols = cols;

    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails.
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    if (rows < 1 || cols < 1){
    	return -1;
    }

    (*mat) = (struct matrix*) malloc(sizeof(struct matrix));
    if ((*mat) == NULL){
    	return -1;
    }
    (*mat)->parent = from;
    (*mat)->data = from->data + offset;
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    from->ref_cnt += 1;
    (*mat)->ref_cnt = 1;

    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    if (mat == NULL){
        return;
    }

    if (mat->parent == NULL){
        if (mat->ref_cnt == 1){
            free(mat->data);
            free(mat);
        } else {
            mat->ref_cnt -= 1;
        }
    } else {
        if ((mat->parent)->ref_cnt == 1){
            deallocate_matrix(mat->parent);
            free(mat);
        } else {
            (mat->parent)->ref_cnt -= 1;
            free(mat);
        }
    }   
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {

    return mat->data[(row * mat->cols) + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    
    mat->data[(row * mat->cols) + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {

    int mr = mat->rows;
    int mc = mat->cols;

    int N = mr * mc;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < (N/16)*16 ; i+=16){

            _mm256_storeu_pd(mat->data + i, _mm256_set1_pd (val));
            _mm256_storeu_pd(mat->data + i + 4, _mm256_set1_pd (val));
            _mm256_storeu_pd(mat->data + i + 8, _mm256_set1_pd (val));
            _mm256_storeu_pd(mat->data + i + 12, _mm256_set1_pd (val));
        }

        #pragma omp for
        for (int i = (N/16)*16; i < N; i++){
            mat->data[i] = val;
        }

    }

}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {

    int mr = mat1->rows;
    int mc = mat1->cols;

    if ((mr != mat2->rows) || (mc != mat2->cols)){
        return -1;
    }

    int N = mr * mc;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < (N/16)*16 ; i+=16) {

            _mm256_storeu_pd(result->data + i, _mm256_add_pd(_mm256_loadu_pd((mat1->data + i)), _mm256_loadu_pd ((mat2->data + i))));
            _mm256_storeu_pd(result->data + i + 4, _mm256_add_pd(_mm256_loadu_pd((mat1->data + i + 4)), _mm256_loadu_pd ((mat2->data + i + 4))));
            _mm256_storeu_pd(result->data + i + 8, _mm256_add_pd(_mm256_loadu_pd((mat1->data + i + 8)), _mm256_loadu_pd ((mat2->data + i + 8))));
            _mm256_storeu_pd(result->data + i + 12, _mm256_add_pd(_mm256_loadu_pd((mat1->data + i + 12)), _mm256_loadu_pd ((mat2->data + i + 12))));
        }

        #pragma omp for
        for (int i = (N/16)*16; i < N; i++){
            result->data[i] = mat1->data[i] + mat2->data[i];
        }
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int mr = mat1->rows;
    int mc = mat1->cols;

    if ((mr != mat2->rows) || (mc != mat2->cols)){
        return -1;
    }

    int N = mr * mc;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < (N/16)*16 ; i+=16){

            _mm256_storeu_pd(result->data + i, _mm256_sub_pd(_mm256_loadu_pd ((mat1->data + i)), _mm256_loadu_pd ((mat2->data + i))));
            _mm256_storeu_pd(result->data + i + 4, _mm256_sub_pd(_mm256_loadu_pd ((mat1->data + i + 4)), _mm256_loadu_pd ((mat2->data + i + 4))));
            _mm256_storeu_pd(result->data + i + 8, _mm256_sub_pd(_mm256_loadu_pd ((mat1->data + i + 8)), _mm256_loadu_pd ((mat2->data + i + 8))));
            _mm256_storeu_pd(result->data + i + 12, _mm256_sub_pd(_mm256_loadu_pd ((mat1->data + i + 12)), _mm256_loadu_pd ((mat2->data + i + 12))));
        }

        #pragma omp for
        for (int i = (N/16)*16; i < N; i++){
            result->data[i] = mat1->data[i] - mat2->data[i];
        }
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.'
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    
    int C = mat2->cols;
    int R2 = mat2->rows;
    int R = mat1->rows;

    if (result->rows != R || result->cols != C || R2 != mat1->cols){
        return -1;
    }

    matrix *transpose;
    allocate_matrix(&transpose, C, R2);


    #pragma omp parallel for
    for (int i = 0; i < (R2 * C)/4 * 4; i+= 4){
        transpose->data[i] = mat2->data[(i/R2) + (i%R2)*C];
        transpose->data[i + 1] = mat2->data[(i + 1)/R2 + (i + 1)%R2 * C];
        transpose->data[i + 2] = mat2->data[(i + 2)/R2 + (i + 2)%R2 * C];
        transpose->data[i + 3] = mat2->data[(i + 3)/R2 + (i + 3)%R2 * C];
    }

    #pragma omp parallel for
    for (int i = (R2 * C)/4 * 4; i < R2 * C; i++){
        transpose->data[i] = mat2->data[(i/R2) + (i%R2)*C];
    }
    
    #pragma omp parallel for
    for (int i = 0; i < R*C; i++){
        double x[4];
        int a = i/C * R2;
        int b = i%C * R2;
        __m256d sum = _mm256_set1_pd (0);
        for (int j = 0; j < (R2/32 * 32); j+= 32){

            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(mat1->data + a + j), _mm256_loadu_pd(transpose->data + b + j)));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(mat1->data + a + j + 4), _mm256_loadu_pd(transpose->data + b + j + 4)));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(mat1->data + a + j + 8), _mm256_loadu_pd(transpose->data + b + j + 8)));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(mat1->data + a + j + 12), _mm256_loadu_pd(transpose->data + b + j + 12)));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(mat1->data + a + j + 16), _mm256_loadu_pd(transpose->data + b + j + 16)));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(mat1->data + a + j + 20), _mm256_loadu_pd(transpose->data + b + j + 20)));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(mat1->data + a + j + 24), _mm256_loadu_pd(transpose->data + b + j + 24)));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(mat1->data + a + j + 28), _mm256_loadu_pd(transpose->data + b + j + 28)));
            
        }

        for (int j = (R2/32 * 32); j < R2/8 * 8; j+= 8){
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(mat1->data + a + j), _mm256_loadu_pd(transpose->data + b + j)));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(mat1->data + a + j + 4), _mm256_loadu_pd(transpose->data + b + j + 4)));
        }

        _mm256_storeu_pd(x, sum);
        for (int j = R2/8 * 8; j < R2; j++){
            x[0] += mat1->data[a + j] * transpose->data[b + j];
        }
        result->data[i] = x[0] + x[1] + x[2] + x[3];
    }

    deallocate_matrix(transpose);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    if (result->rows != mat->rows || result->cols != mat->cols || mat->cols != mat->rows || pow < 0){
        return -1;
    }

    if (pow == 0){
        fill_matrix(result, 0);
        #pragma omp parallel for
        for (int i = 0; i < result->rows; i++){
            result->data[(i * result->cols) + i] = 1; 
        }
        return 0;
    }

    if (pow == 1){
        memcpy(result->data, mat->data, sizeof(double) * result->rows * result->cols);
        return 0;
    }

    matrix* t;
    allocate_matrix(&t, mat->rows, mat->cols);

    matrix* e;
    allocate_matrix(&e, mat->rows, mat->cols);

    mul_matrix(t, mat, mat);
    memcpy(e->data, t->data, sizeof(double) * result->rows * result->cols);

    if (pow%2 == 0){    
        pow_matrix(result, e, pow/2);
    } else {
        pow_matrix(t, e, (pow - 1)/2);
        mul_matrix(result, t, mat);
    }

    deallocate_matrix(t);
    deallocate_matrix(e);
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    int mr = mat->rows;
    int mc = mat->cols;

    if ((mr != result->rows) || (mc != result->cols)){
        return -1;
    }

    int N = mr * mc;
    __m256d y = _mm256_set1_pd((double) 0);

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < (N/16)*16 ; i+=16){

            _mm256_storeu_pd(result->data + i, _mm256_sub_pd(y, _mm256_loadu_pd ((mat->data + i))));
            _mm256_storeu_pd(result->data + i + 4, _mm256_sub_pd(y, _mm256_loadu_pd ((mat->data + i + 4))));
            _mm256_storeu_pd(result->data + i + 8, _mm256_sub_pd(y, _mm256_loadu_pd ((mat->data + i + 8))));
            _mm256_storeu_pd(result->data + i + 12, _mm256_sub_pd(y, _mm256_loadu_pd ((mat->data + i + 12))));
        }

        #pragma omp for
        for (int i = (N/16)*16; i < N; i++){
            result->data[i] = -mat->data[i];
        }
    }

    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    int mr = mat->rows;
    int mc = mat->cols;

    if ((mr != result->rows) || (mc != result->cols)){
        return -1;
    }

    int N = mr * mc;
    __m256d y = _mm256_set1_pd((double) 0);

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < (N/16)*16 ; i+=16){

            _mm256_storeu_pd(result->data + i,
             _mm256_max_pd(_mm256_loadu_pd (mat->data + i), _mm256_sub_pd(y, _mm256_loadu_pd ((mat->data + i)))));
            _mm256_storeu_pd(result->data + i + 4,
             _mm256_max_pd(_mm256_loadu_pd (mat->data + i + 4), _mm256_sub_pd(y, _mm256_loadu_pd ((mat->data + i + 4)))));
            _mm256_storeu_pd(result->data + i + 8,
             _mm256_max_pd(_mm256_loadu_pd (mat->data + i + 8), _mm256_sub_pd(y, _mm256_loadu_pd ((mat->data + i + 8)))));
            _mm256_storeu_pd(result->data + i + 12,
             _mm256_max_pd(_mm256_loadu_pd (mat->data + i + 12), _mm256_sub_pd(y, _mm256_loadu_pd ((mat->data + i + 12)))));
        }

        #pragma omp for
        for (int i = (N/16)*16; i < N; i++){
            result->data[i] = (mat->data[i] > 0)? mat->data[i]: -mat->data[i];
        }
    }

    return 0;
}

