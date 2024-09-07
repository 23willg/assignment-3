__kernel void matrixMultiply(
    __global const float *A,  __global const float *B,  __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns)
{
  //@@ Insert code to implement matrix multiplication here

  const int tile_size = 16; 
  __local float A_tile[tile_size][tile_size];
  __local float B_tile[tile_size][tile_size];

  // local rows and col
  unsigned int row_loc = get_local_id(0);
  unsigned int col_loc = get_local_id(1);

  // absolute row and col
  unsigned int row = (tile_size * get_group_id(0)) + row_loc;
  unsigned int col = (tile_size * get_group_id(1)) + col_loc;

  //printf("row: %d, row)
  //printf("col: %d\n", col);
  //printf("row_loc: %d\n", row_loc);
  //printf("col_loc: %d\n", col_loc);
  //printf("group1: %d\n", get_group_id(0));
  //printf("group2: %d\n", get_group_id(1));

  float accum = 0;

  unsigned int num_tiles = ((numAColumns + tile_size) / tile_size);

  //loops through the tiles
  for (unsigned int cur_tile = 0; cur_tile < num_tiles; cur_tile++)
  {
    //loops within tiles
    for (unsigned int i = 0; i < tile_size; i++)
    {

      // (cur_tile * tile_size) is the beginning boundary of the tile
      unsigned int tile_offset = (cur_tile * tile_size) + i;

      if ((row < numARows) && (tile_offset < numAColumns))
      {
        A_tile[row_loc][i] = A[row * numAColumns + tile_offset];
        //printf("A%f\n", A[row * numAColumns + tile_offset]);
      }
      else
      {
        A_tile[row_loc][i] = 0;
      }

      if ((tile_offset < numBRows) && (col < numBColumns))
      {
        B_tile[i][col_loc] = B[tile_offset * numBColumns + col];
        //printf("B%f\n", B[tile_offset * numBColumns + col]);
      }
      else
      {
        B_tile[i][col_loc] = 0;
      }

    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int i = 0; i < tile_size; i++)
    {
      accum += (A_tile[row_loc][i] * B_tile[i][col_loc]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if ((row < numCRows) && (col < numCColumns))
  {
    C[row * numCColumns + col] = accum;
  }
}