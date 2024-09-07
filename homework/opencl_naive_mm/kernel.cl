__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here

  unsigned int col = get_global_id(0) % numCColumns;
  unsigned int row = get_global_id(0) / numCColumns;

  for (unsigned int i = 0; i < numAColumns; i++)
  {
    C[row * numCColumns + col] += (A[row * numAColumns + i] * B[i * numBColumns + col]);
  }


}
