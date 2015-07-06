/************************************************************************/
/*2012-3 Sheng Yang
/*矩阵基本操作 CUDA 程序 核函数实现 和线程块划分等
/************************************************************************/

#include <cuda_runtime.h>
#include <cutil.h>
#include <iostream>
#include <cutil_inline.h>
#include <cufft.h>

using namespace std;


#define MIN_DATA -999999999 //求最大值时用到的最小值
#define BLOCK_SIZE 512		//一维划分块大小
#define MUL_BLOCK_SIZE 16	//二维划分块大小
#define KAHAN_SUMMATION_FORMULA 0 //是否利用卡亨公式减少求和过程中的误差累积

//矩阵按行求最大值
//基本思想: 每一个Block负责一行 ，基本满足合并访问
//下面按行操作的运算 也是同一思想 不再赘述
//利用shared memory进行并行归约式加法：
/************************************************************************/
/* A-------A------A-------A
/*   \   /           \  /
/*     2A             2A
/*       \           /
/*	           4A
/************************************************************************/
//以下提到归约算法均是这种思想
//@param data_in	输入 矩阵
//@param row		输入 矩阵行
//@param column		输入 矩阵列
//@param index		输出 最大值所在位置
//@param value		输出 最大值
__global__ void Matrix_MaxofRow(float *data_in, int row, int column, int *index, float *value)
{

	__shared__ float max[BLOCK_SIZE];
	__shared__ float ind[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;

	max[tid] = MIN_DATA;
	ind[tid] = -1;

	if( tid >= column || bid >= row)
		return;

	unsigned int begin_addr, end_addr;
	begin_addr = bid * column;
	end_addr = begin_addr + column;
	begin_addr += tid;

	while(begin_addr < end_addr)
	{
		float temp = data_in[begin_addr];
		if(max[tid] < temp)
		{
			max[tid] = temp;
			ind[tid] = begin_addr;
		}
		begin_addr += blockDim.x;
	}
	__syncthreads();

	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
	{
		if(tid < s)
		{
			if(max[tid] < max[tid+s])
			{
				max[tid] = max[tid+s];
				ind[tid] = ind[tid+s];
			}
		}
		__syncthreads();
	}

	if(tid == 0)
	{
		value[bid] = max[0];
		index[bid] = ind[0];
	}

}
//矩阵按列求最大值
//基本思想：为了满足合并访问的要求 每个块负责了128(其实可以更小)列的运算
//下面按列的算法都只对BLOCK_SIZE = 512适用 改变了BLOCK_SZIE大小的话 要更改核函数
//针对每一个128列 若线程ID = ida 
//则线程ID为 ida%,(ida+128)%512,(ida+256)%512 (ida+384)%512 负责同一列的数据运算
//最后汇总一个块就可以求得128列的每列的最大值 以下按列求和也是这样的思想 不再赘述
//@param data_in	输入 矩阵
//@param row		输入 矩阵行
//@param column		输入 矩阵列
//@param index		输出 最大值所在位置
//@param value		输出 最大值
__global__ void Matrix_MaxofCol(float *data_in, int row, int column, int *index, float *value)
{
	__shared__ float max[BLOCK_SIZE];
	__shared__ int ind[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;

	max[tid] = MIN_DATA;
	ind[tid] = -1;
	int res = column - (bid + 1)*128;
	unsigned int begin_addr = 128*bid + (tid>>7) * column + (tid&127);
	unsigned int end_addr = row * column;

	if( (tid&127) >= (res + 128))
		return;

	while(begin_addr < end_addr )
	{
		float temp = data_in[begin_addr];
		if(max[tid] < temp)
		{
			max[tid] = temp;
			ind[tid] = begin_addr;
		}
		begin_addr += 4* column;
	}
	__syncthreads();
	if(tid < 256)
	{
		if(max[tid] < max[tid + 256])
		{
			max[tid] = max[tid + 256];
			ind[tid] = ind[tid + 256];
		}
	}
	__syncthreads();

	if(tid < 128)
	{
		if(max[tid] < max[tid + 128])
		{
			max[tid] = max[tid + 128];
			ind[tid] = ind[tid + 128];
		}
	}
	__syncthreads();

	if(tid < res+128 && tid < 128){
		value[tid + bid * 128] = max[tid];
		index[tid + bid * 128] = ind[tid];
	}

}

//实数按行求和
//@param data_in	输入 矩阵
//@param row		输入 矩阵行
//@param column		输入 矩阵列
//@param value		输出 和
__global__ void Matrix_SumofRow(float *data_in, int row, int column, float *value)
{
	__shared__ float sum[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;

	sum[tid] = 0;

	if( tid >= column || bid >= row)
		return;

	unsigned int begin_addr, end_addr;

	begin_addr = bid * column;
	end_addr = begin_addr + column;
	begin_addr += tid;

	while(begin_addr < end_addr)
	{
		sum[tid] += data_in[begin_addr];
		begin_addr += blockDim.x;
	}

	__syncthreads();

	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
	{
		if(tid < s)
		{
			sum[tid] += sum[tid + s];
		}

		__syncthreads();
	}

	if(tid == 0)
		value[bid] = sum[0];

}

//实数按列求和
//@param data_in	输入 矩阵
//@param row		输入 矩阵行
//@param column		输入 矩阵列
//@param value		输出 和
__global__ void Matrix_SumofCol(float *data_in, int row, int column, float *value)
{
	__shared__ float sum[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;

	sum[tid] = 0;
	int res = column - (bid + 1)*128;
	unsigned int begin_addr = 128*bid + (tid>>7) * column + (tid&127);
	unsigned int end_addr = row * column;

	if( (tid&127) >= (res + 128))
		return;

	while(begin_addr < end_addr )
	{
		sum[tid] += data_in[begin_addr];
		begin_addr += 4* column;
	}

	__syncthreads();
	if(tid < 256)
		sum[tid] += sum[tid + 256];

	__syncthreads();

	if(tid < 128)
		sum[tid] += sum[tid + 128];
	__syncthreads();

	if(tid < res+128 && tid < 128)
		value[tid + bid * 128] = sum[tid];



}

//实数矩阵总和 
//这里其实先对矩阵进行了按行求和操作 
//最后实现的是一个块对一位数据的归约求和操作
//@param temp_value		输入数据
//@param num			数据个数
//@param value			输出和	
__global__ void Matrix_SumofAll(float *temp_value, int num, float *value)
{
	__shared__ float sum[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;

	sum[tid] = 0;

	if( tid >= num)
		return;

	unsigned int begin_addr, end_addr;

	begin_addr = 0;
	end_addr = num;
	begin_addr += tid;

	while(begin_addr < end_addr)
	{
		sum[tid] += temp_value[begin_addr];
		begin_addr += blockDim.x;
	}

	__syncthreads();

	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
	{
		if(tid < s)
		{
			sum[tid] += sum[tid + s];
		}

		__syncthreads();
	}
	if(tid == 0)
		value[0] = sum[0];
}

//复数按行求和
//@param data_in	输入 矩阵
//@param row		输入 矩阵行
//@param column		输入 矩阵列
//@param value		输出 和
__global__ void Matrix_SumofRow(float2 *data_in, int row, int column, float2 *value)
{

	__shared__ float sum_R[BLOCK_SIZE];
	__shared__ float sum_I[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;

	sum_R[tid] = 0;
	sum_I[tid] = 0;

	if( tid >= column || bid >= row)
		return;

	unsigned int begin_addr, end_addr;

	begin_addr = bid * column;
	end_addr = begin_addr + column;
	begin_addr += tid;

	while(begin_addr < end_addr)
	{	

		float2 temp = data_in[begin_addr];
		sum_R[tid] += temp.x;
		sum_I[tid] += temp.y;
		begin_addr += blockDim.x;
	}

	__syncthreads();

	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
	{
		if(tid < s)
		{
			sum_R[tid] += sum_R[tid + s];
			sum_I[tid] += sum_I[tid + s];

		}

		__syncthreads();
	}

	if(tid == 0)
	{
		float2 temp;
		temp.x = sum_R[0];
		temp.y = sum_I[0];
		value[bid] = temp;
	}

}

//复数按列求和
//@param data_in	输入 矩阵
//@param row		输入 矩阵行
//@param column		输入 矩阵列
//@param value		输出 和
__global__ void Matrix_SumofCol(float2 *data_in, int row, int column, float2 *value)
{
	__shared__ float sum_I[BLOCK_SIZE];
	__shared__ float sum_R[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;

	sum_I[tid] = 0;
	sum_R[tid] = 0;
	int res = column - (bid + 1)*128;
	unsigned int begin_addr = 128*bid + (tid>>7) * column + (tid&127);
	unsigned int end_addr = row * column;

	if( (tid&127) >= (res + 128))
		return;

	while(begin_addr < end_addr )
	{
		float2 temp = data_in[begin_addr];
		sum_R[tid] += temp.x;
		sum_I[tid] += temp.y;
		begin_addr += 4* column;
	}

	__syncthreads();
	if(tid < 256){
		sum_R[tid] += sum_R[tid + 256];
		sum_I[tid] += sum_I[tid + 256];

	}

	__syncthreads();

	if(tid < 128){
		sum_R[tid] += sum_R[tid + 128];
		sum_I[tid] += sum_I[tid + 128];

	}
	__syncthreads();

	if(tid < res+128 && tid < 128){
		float2 temp;
		temp.x = sum_R[tid];
		temp.y = sum_I[tid];
		value[tid + bid * 128] = temp;
	}
}

//复数求总和
//@param data_in	输入 矩阵
//@param row		输入 矩阵行
//@param column		输入 矩阵列
//@param value		输出 和
__global__ void Matrix_SumofAll(float2 *temp_value, int num, float2 *value)
{
	__shared__ float sum_I[BLOCK_SIZE];
	__shared__ float sum_R[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;

	sum_R[tid] = 0;
	sum_I[tid] = 0;

	if( tid >= num)
		return;

	unsigned int begin_addr, end_addr;

	begin_addr = 0;
	end_addr = num;
	begin_addr += tid;

	while(begin_addr < end_addr)
	{
		float2 temp = temp_value[begin_addr];
		sum_R[tid] += temp.x;
		sum_I[tid] += temp.y;
		begin_addr += blockDim.x;
	}

	__syncthreads();

	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
	{
		if(tid < s)
		{
			sum_R[tid] += sum_R[tid + s];
			sum_I[tid] += sum_I[tid + s];
		}

		__syncthreads();
	}
	if(tid == 0){
		float2 temp;
		temp.x = sum_R[0];
		temp.y = sum_I[0];
		value[0] = temp;
	}
}

//实数矩阵乘法
//基本思想：每个块负责一个输出矩阵中16*16小块的结果运算 
//利用shared memory来实现对global memory的合并访问和存储
//@param Matrix_a	输入 矩阵A
//@param row_a		输入 矩阵A行
//@param col_a		输入 矩阵A列
//@param lda		输入 矩阵A在显存中实际的每行大小
//@param Matrix_b	输入 矩阵B
//@param row_b		输入 矩阵B行
//@param col_b		输入 矩阵B列
//@param ldb		输入 矩阵B在显存中实际的每行大小
//@param Matrix_c	输出 矩阵 A*B
//@param ldc		输入 矩阵C在显存中实际的每行大小
__global__ static void Mult_kernel(	const float *Matrix_a, int row_a, int col_a, size_t lda,
								   const float *Matrix_b, int row_b, int col_b, size_t ldb,
								   float *Matrix_c, size_t ldc)
{
	__shared__ float matrix_a[MUL_BLOCK_SIZE][MUL_BLOCK_SIZE];
	__shared__ float matrix_b[MUL_BLOCK_SIZE][MUL_BLOCK_SIZE];

	unsigned int tidr = threadIdx.x;
	unsigned int tidc = threadIdx.y;
	unsigned int bidr = blockIdx.x * MUL_BLOCK_SIZE;
	unsigned int bidc = blockIdx.y * MUL_BLOCK_SIZE;

	int i,j;
	float result = 0;
	float comp = 0;

	for(j = 0; j < col_a; j += MUL_BLOCK_SIZE)
	{
		if(tidr + bidr < row_a && tidc + j < col_a)
			matrix_a[tidr][tidc] = Matrix_a[(tidr + bidr) * lda + tidc + j];
		else
			matrix_a[tidr][tidc] = 0;
		if(tidr + j < row_b && tidc + bidc < col_b)
			matrix_b[tidr][tidc] = Matrix_b[(tidr + j) * ldb + tidc + bidc];
		else
			matrix_b[tidr][tidc] = 0;

		__syncthreads();
		if(!KAHAN_SUMMATION_FORMULA)
			for(i = 0; i < MUL_BLOCK_SIZE; i++)
			{
				result += matrix_a[tidr][i] * matrix_b[i][tidc];
			}
		else
			for(i = 0; i < MUL_BLOCK_SIZE; i++)
			{
				float t;
				comp -= matrix_a[tidr][i] * matrix_b[i][tidc];
				t = result - comp;
				comp = (t - result) + comp;
				result = t;
			}

			__syncthreads();
	}
	if(tidr + bidr < row_a)
		Matrix_c[(tidr + bidr) * ldc + tidc + bidc] = result;
}

//复数矩阵乘法
//基本思想：每个块负责一个输出矩阵中16*16小块的结果运算 
//利用shared memory来实现对global memory的合并访问和存储
//@param Matrix_a	输入 矩阵A
//@param row_a		输入 矩阵A行
//@param col_a		输入 矩阵A列
//@param lda		输入 矩阵A在显存中实际的每行大小
//@param Matrix_b	输入 矩阵B
//@param row_b		输入 矩阵B行
//@param col_b		输入 矩阵B列
//@param ldb		输入 矩阵B在显存中实际的每行大小
//@param Matrix_c	输出 矩阵 A*B
//@param ldc		输入 矩阵C在显存中实际的每行大小
__global__ static void Mult_kernel(	const float2 *Matrix_a, int row_a, int col_a, size_t lda,
								   const float2 *Matrix_b, int row_b, int col_b, size_t ldb,
								   float2 *Matrix_c, size_t ldc)
{
	__shared__ float2 matrix_a[MUL_BLOCK_SIZE][MUL_BLOCK_SIZE];
	__shared__ float2 matrix_b[MUL_BLOCK_SIZE][MUL_BLOCK_SIZE];

	unsigned int tidr = threadIdx.x;
	unsigned int tidc = threadIdx.y;
	unsigned int bidr = blockIdx.x * MUL_BLOCK_SIZE;
	unsigned int bidc = blockIdx.y * MUL_BLOCK_SIZE;

	int i,j;
	float2 result = make_float2(0,0);
	float2 comp = make_float2(0,0);

	for(j = 0; j < col_a; j += MUL_BLOCK_SIZE)
	{
		if(tidr + bidr < row_a && tidc + j < col_a)
			matrix_a[tidr][tidc] = Matrix_a[(tidr + bidr) * lda + tidc + j];
		else
			matrix_a[tidr][tidc] = make_float2(0,0);
		if(tidr + j < row_b && tidc + bidc < col_b)
			matrix_b[tidr][tidc] = Matrix_b[(tidr + j) * ldb + tidc + bidc];
		else
			matrix_b[tidr][tidc] = make_float2(0,0);

		__syncthreads();

		if(!KAHAN_SUMMATION_FORMULA)
			for(i = 0; i < MUL_BLOCK_SIZE; i++)
			{
				result.x+= matrix_a[tidr][i].x * matrix_b[i][tidc].x;
				result.y+= matrix_a[tidr][i].y * matrix_b[i][tidc].y;
			}
		else
			for(i = 0; i < MUL_BLOCK_SIZE; i++)
			{
				float2 t;
				comp.x -= matrix_a[tidr][i].x * matrix_b[i][tidc].x;
				comp.y -= matrix_a[tidr][i].y * matrix_b[i][tidc].y;
				t.x = result.x - comp.x;
				t.y = result.y - comp.y;
				comp.x = (t.x - result.x) + comp.x;
				comp.y = (t.y - result.y) + comp.y;
				result = t;
			}


			__syncthreads();
	}
	if(tidr + bidr < row_a)
		Matrix_c[(tidr + bidr) * ldc + tidc + bidc] = result;
}

//矩阵转置
//基本思想:利用sharedmemory实现小块16*16的转置 实现读取写入的合并
//@param T					输入 矩阵数据类型
//@param Matrix_in		输入 矩阵数据
//@param row				输入	矩阵行数
//@param col				输入 矩阵列数
//@param Matrix_out	输出	转置后的矩阵
template<class T>	
__global__ static void Transpose_kernel(const T * Matrix_in, int row, int col, T * Matrix_out)
{
	__shared__ T temp[MUL_BLOCK_SIZE][MUL_BLOCK_SIZE + 1];

	unsigned int xIndex = blockIdx.x * MUL_BLOCK_SIZE + threadIdx.x;
	unsigned int yIndex = blockIdx.y * MUL_BLOCK_SIZE + threadIdx.y;

	if((xIndex < col) && (yIndex < row))
	{
		unsigned int index_in = yIndex * col + xIndex;
		temp[threadIdx.y][threadIdx.x] = Matrix_in[index_in];
	}

	__syncthreads();

	xIndex = MUL_BLOCK_SIZE*blockIdx.y + threadIdx.x;
	yIndex = MUL_BLOCK_SIZE*blockIdx.x + threadIdx.y;

	if((xIndex < row) && (yIndex < col))
	{
		unsigned int index_out = yIndex * row + xIndex;
		Matrix_out[index_out] = temp[threadIdx.x][threadIdx.y];
	}

}

/************************************************************************/
/* 矩阵求最大值和最大值所在位置 flag = 0 按行求最大值 flag = 1 按列求最大值
/* @param data_in	:	输入 矩阵数据 内存数据 一维连续存储
/* @param row		:	输入 矩阵行数
/* @param col		:	输入 矩阵列数
/* @param index		:	输出 表示每行/每列最大值所在位置
/* @param value		:	输出 表示每行/每列的最大值
/* @param flag		:	输入 0 表示按行操作 1 表示按列操作
/************************************************************************/
extern "C"
void Matrix_Max(float *data_in, int row, int col, 
				int *index, float *value, int flag)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float memory;

	float *d_data;
	int *d_index;
	float *d_value;
	int res_num = 0;

	if(flag == 0)
		res_num = row;
	else
		res_num = col;

	CUDA_SAFE_CALL(cudaMalloc((void**)& d_data, sizeof(float)* row * col));
	CUDA_SAFE_CALL(cudaMalloc((void**)& d_index, sizeof(int)* res_num));
	CUDA_SAFE_CALL(cudaMalloc((void**)& d_value, sizeof(float)* res_num));

	memory = sizeof(float)*row*col + sizeof(int)*res_num + sizeof(float)*res_num;

	CUDA_SAFE_CALL(cudaMemcpy(d_data, data_in,sizeof(float)*row*col, cudaMemcpyHostToDevice));

	if(flag == 0)
	{
		dim3 block_size(BLOCK_SIZE,1,1);
		dim3 grid_size(row,1,1);

		Matrix_MaxofRow<<<grid_size,block_size>>>(d_data, row,  col,d_index, d_value);

		cutilCheckMsg("kernel launch failure");

	}
	else
	{
		dim3 block_size(BLOCK_SIZE,1,1);
		dim3 gird_size(col/128+1,1,1);

		Matrix_MaxofCol<<<gird_size, block_size>>>(d_data, row, col, d_index, d_value);

		cutilCheckMsg("kernel launch failure");
	}

	CUDA_SAFE_CALL(cudaMemcpy(index, d_index, res_num * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(value, d_value, res_num * sizeof(float), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(d_data));
	CUDA_SAFE_CALL(cudaFree(d_index));
	CUDA_SAFE_CALL(cudaFree(d_value));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout<<"函数运行时间:	"<<time<<" ms"<<std::endl;
	cout<<"函数使用显存:	"<<memory/1024<<" KB"<<endl;
}

/************************************************************************/
/*实数矩阵求和 flag = 0 按行求和 flag = 1 按列求和 flag = 2 全部求和
/*@param data_in	:	输入 矩阵数据 内存数据 一维连续存储
/*@param row		:	输入 矩阵行数
/*@param col		:	输入 矩阵列数
/*@param value		:	输出 每行/列求得的和
/*@param flag		:	输入 0 按行操作 1 按列操作 2 全部数操作
/************************************************************************/
extern "C"
void Matrix_Sum(float *data_in, int row, int col, float *value, int flag)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float memory;

	float *d_data;
	float *d_value;
	int res_num = 0;

	if(flag == 0)
		res_num = row;
	else if(flag == 1)
		res_num = col;
	else
		res_num = 1;

	CUDA_SAFE_CALL(cudaMalloc((void**)& d_data, sizeof(float)* row * col));
	CUDA_SAFE_CALL(cudaMalloc((void**)& d_value, sizeof(float)* res_num));

	memory = sizeof(float)*row*col + sizeof(float)*res_num;

	CUDA_SAFE_CALL(cudaMemcpy(d_data, data_in,sizeof(float)*row*col, cudaMemcpyHostToDevice));

	if(flag == 0)
	{
		dim3 block_size(BLOCK_SIZE,1,1);
		dim3 grid_size(row,1,1);

		Matrix_SumofRow<<<grid_size,block_size>>>(d_data, row,  col, d_value);

		cutilCheckMsg("kernel launch failure");

	}
	else if(flag == 1)
	{
		dim3 block_size(BLOCK_SIZE,1,1);
		dim3 gird_size(col/128+1,1,1);

		Matrix_SumofCol<<<gird_size, block_size>>>(d_data, row, col, d_value);

		cutilCheckMsg("kernel launch failure");
	}
	else
	{
		dim3 block_size(BLOCK_SIZE,1,1);
		dim3 gird_size(row,1,1);
		float *temp_value;
		cudaMalloc((void**)&temp_value, sizeof(float)* row);
		memory += sizeof(float)*row;

		Matrix_SumofRow<<<gird_size, block_size>>>(d_data, row, col, temp_value);
		cutilCheckMsg("kernel launch failure");
		Matrix_SumofAll<<<1,block_size>>>(temp_value,row,d_value);
		cutilCheckMsg("kernel launch failure");
	}

	CUDA_SAFE_CALL(cudaMemcpy(value, d_value, res_num * sizeof(float), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(d_data));
	CUDA_SAFE_CALL(cudaFree(d_value));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout<<"函数运行时间:	"<<time<<" ms"<<std::endl;
	cout<<"函数使用显存:	"<<memory/1024<<" KB"<<endl;

}

/************************************************************************/
/*复数矩阵求和 flag = 0 按行求和 flag = 1 按列求和 flag = 2 全部求和
/*@param data_in	:	输入 矩阵数据 内存数据 一维连续存储
/*@param row		:	输入 矩阵行数
/*@param col		:	输入 矩阵列数
/*@param value		:	输出 每行/列求得的和
/*@param flag		:	输入 0 按行操作 1 按列操作 2 全部数操作
/************************************************************************/
extern "C"
void Matrix_SumCom(cuComplex *data_in, int row, int col, cuComplex *value, int flag)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float memory;

	float2 *d_data;
	float2 *d_value;
	int res_num = 0;

	if(flag == 0)
		res_num = row;
	else if(flag == 1)
		res_num = col;
	else
		res_num = 1;

	CUDA_SAFE_CALL(cudaMalloc((void**)& d_data, sizeof(float2)* row * col));
	CUDA_SAFE_CALL(cudaMalloc((void**)& d_value, sizeof(float2)* res_num));

	memory = sizeof(float2)*row*col + sizeof(float2)*res_num;


	CUDA_SAFE_CALL(cudaMemcpy(d_data, data_in,sizeof(float2)*row*col, cudaMemcpyHostToDevice));

	if(flag == 0)
	{
		dim3 block_size(BLOCK_SIZE,1,1);
		dim3 grid_size(row,1,1);

		Matrix_SumofRow<<<grid_size,block_size>>>(d_data, row,  col, d_value);

		cutilCheckMsg("kernel launch failure");

	}
	else if(flag == 1)
	{
		dim3 block_size(BLOCK_SIZE,1,1);
		dim3 gird_size(col/128+1,1,1);

		Matrix_SumofCol<<<gird_size, block_size>>>(d_data, row, col, d_value);

		cutilCheckMsg("kernel launch failure");
	}
	else
	{
		dim3 block_size(BLOCK_SIZE,1,1);
		dim3 gird_size(row,1,1);
		float2 *temp_value;
		cudaMalloc((void**)&temp_value, sizeof(float2)* row);
		memory += sizeof(float2) * row;

		Matrix_SumofRow<<<gird_size, block_size>>>(d_data, row, col, temp_value);
		cutilCheckMsg("kernel launch failure");
		Matrix_SumofAll<<<1,block_size>>>(temp_value,row,d_value);
		cutilCheckMsg("kernel launch failure");
	}

	CUDA_SAFE_CALL(cudaMemcpy(value, d_value, res_num * sizeof(float2), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(d_data));
	CUDA_SAFE_CALL(cudaFree(d_value));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout<<"函数运行时间:	"<<time<<" ms"<<std::endl;
	cout<<"函数使用显存:	"<<memory/1024<<" KB"<<endl;
}

/************************************************************************/
/*实数矩阵乘法 
/*@param Matrix_a	:	输入 矩阵A数据 内存数据 一维连续存储
/*@param row_a		:	输入 矩阵A行数
/*@param col_a		:	输入 矩阵A列数
/*@param Matrix_b	:	输入 矩阵B数据 内存数据 一维连续存储
/*@param row_b		:	输入 矩阵B行数
/*@param col_b		:	输入 矩阵B列数
/*@param Matrix_c	:	输出 矩阵C数据
/*@param flag		:	输出 0 数据正常 1  相乘矩阵行列部匹配
/************************************************************************/
extern "C"
void Matrix_Multi(const float *Matrix_a, int row_a, int col_a, const float *Matrix_b, int row_b, int col_b, float *Matrix_c)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float memory = 0;

	float *Matrix_da, *Matrix_db, *Matrix_dc;

	dim3 threads(MUL_BLOCK_SIZE,MUL_BLOCK_SIZE);
	int block_width = (row_a + MUL_BLOCK_SIZE -1)/MUL_BLOCK_SIZE;
	int block_height = (col_b + MUL_BLOCK_SIZE -1)/MUL_BLOCK_SIZE;

	dim3 blocks(block_width, block_height);

	//申请显存操作 为了方便16*16块的乘法操作 使用cudaMallocPitch制定对齐
	//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
	//dst 复制目标地址 src 复制源地址
	//dpitch 是指 复制到目标地址的实际的每行数据大小(byte)
	//spitch 是指 在源地址中 复制到相应行的实际地址大小(byte)
	//width  是指 复制到目标地址的每行的宽度(byte)
	//height 是指 复制到目标地址的行数(个)
	//kind	 是指 复制类型了

	size_t pitch_a, pitch_b, pitch_c;

	//为了对齐和使得行是16的倍数 配置申请的显存对齐

	CUDA_SAFE_CALL(cudaMallocPitch((void**) & Matrix_da, &pitch_a, sizeof(float) * col_a, row_a));
	CUDA_SAFE_CALL(cudaMallocPitch((void**) & Matrix_db, &pitch_b, sizeof(float) * col_b, row_b));
	CUDA_SAFE_CALL(cudaMallocPitch((void**) & Matrix_dc, &pitch_c, sizeof(float) * col_b, row_a));

	memory += pitch_a * row_a + pitch_b * row_b + pitch_c * row_a;

	/*cout<<"Pitch_a: "<<pitch_a<<endl;
	cout<<"Pitch_b: "<<pitch_b<<endl;
	cout<<"Pitch_c: "<<pitch_c<<endl*/;

	CUDA_SAFE_CALL(cudaMemcpy2D(	Matrix_da, pitch_a, Matrix_a, sizeof(float)*col_a,
		sizeof(float)*col_a, row_a, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy2D(	Matrix_db, pitch_b, Matrix_b, sizeof(float)*col_b,
		sizeof(float)*col_b, row_b, cudaMemcpyHostToDevice));

	//Kernel

	Mult_kernel<<<blocks,threads>>>(Matrix_da,row_a,col_a,pitch_a/sizeof(float),
		Matrix_db,row_b,col_b,pitch_b/sizeof(float),
		Matrix_dc,pitch_c/sizeof(float));

	cutilCheckMsg("kernel launch failure");

	//复制运算结果到主机端
	CUDA_SAFE_CALL(cudaMemcpy2D(	
		Matrix_c, sizeof(float)*col_b, Matrix_dc, pitch_c,
		sizeof(float)*col_b, row_a, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(Matrix_da));
	CUDA_SAFE_CALL(cudaFree(Matrix_db));
	CUDA_SAFE_CALL(cudaFree(Matrix_dc));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cout<<"函数运行时间:	"<<time<<" ms"<<std::endl;
	cout<<"函数使用显存:	"<<memory/1024<<" KB"<<endl;
}

/************************************************************************/
/*复数矩阵乘法 
/*@param Matrix_a	:	输入 矩阵A数据 内存数据 一维连续存储
/*@param row_a		:	输入 矩阵A行数
/*@param col_a		:	输入 矩阵A列数
/*@param Matrix_b	:	输入 矩阵B数据 内存数据 一维连续存储
/*@param row_b		:	输入 矩阵B行数
/*@param col_b		:	输入 矩阵B列数
/*@param Matrix_c	:	输出 矩阵C数据
/************************************************************************/
extern "C"
void Matrix_MultiCom(const float2 *Matrix_a, int row_a, int col_a, const float2 *Matrix_b, int row_b, int col_b, float2 *Matrix_c)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float memory = 0;

	float2 *Matrix_da, *Matrix_db, *Matrix_dc;

	dim3 threads(MUL_BLOCK_SIZE,MUL_BLOCK_SIZE);
	int block_width = (row_a + MUL_BLOCK_SIZE -1)/MUL_BLOCK_SIZE;
	int block_height = (col_b + MUL_BLOCK_SIZE -1)/MUL_BLOCK_SIZE;

	dim3 blocks(block_width, block_height);

	//申请显存操作 为了方便16*16块的乘法操作 使用cudaMallocPitch制定对齐

	//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
	//dst 复制目标地址 src 复制源地址
	//dpitch 是指 复制到目标地址的实际的每行数据大小(byte)
	//spitch 是指 在源地址中 复制到相应行的实际地址大小(byte)
	//width  是指 复制到目标地址的每行的宽度(byte)
	//height 是指 复制到目标地址的行数(个)
	//kind	 是指 复制类型了

	size_t pitch_a, pitch_b, pitch_c;

	//为了对齐和使得行是16的倍数 配置申请的显存对齐

	CUDA_SAFE_CALL(cudaMallocPitch((void**) & Matrix_da, &pitch_a, sizeof(float2) * col_a, row_a));
	CUDA_SAFE_CALL(cudaMallocPitch((void**) & Matrix_db, &pitch_b, sizeof(float2) * col_b, row_b));
	CUDA_SAFE_CALL(cudaMallocPitch((void**) & Matrix_dc, &pitch_c, sizeof(float2) * col_b, row_a));

	memory += pitch_a * row_a + pitch_b * row_b + pitch_c * row_a;

	//cout<<"Pitch_a: "<<pitch_a<<endl;
	//cout<<"Pitch_b: "<<pitch_b<<endl;
	//cout<<"Pitch_c: "<<pitch_c<<endl;

	CUDA_SAFE_CALL(cudaMemcpy2D(	Matrix_da, pitch_a, Matrix_a, sizeof(float2)*col_a,
		sizeof(float2)*col_a, row_a, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy2D(	Matrix_db, pitch_b, Matrix_b, sizeof(float2)*col_b,
		sizeof(float2)*col_b, row_b, cudaMemcpyHostToDevice));

	//Kernel

	Mult_kernel<<<blocks,threads>>>(Matrix_da,row_a,col_a,pitch_a/sizeof(float2),
		Matrix_db,row_b,col_b,pitch_b/sizeof(float2),
		Matrix_dc,pitch_c/sizeof(float2));

	cutilCheckMsg("kernel launch failure");

	CUDA_SAFE_CALL(cudaMemcpy2D(	
		Matrix_c, sizeof(float2)*col_b, Matrix_dc, pitch_c,
		sizeof(float2)*col_b, row_a, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(Matrix_da));
	CUDA_SAFE_CALL(cudaFree(Matrix_db));
	CUDA_SAFE_CALL(cudaFree(Matrix_dc));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cout<<"函数运行时间:	"<<time<<" ms"<<std::endl;
	cout<<"函数使用显存:	"<<memory/1024<<" KB"<<endl;
}

/************************************************************************/
/*实数矩阵转置
/*@param Matrix_in :	输入 需要转置的矩阵 内存数据 一维连续存储
/*@param row			:	输入 矩阵行数
/*@param col			:	输入 矩阵列数
/*@param Matrix_out:	输出	转置后的矩阵
/************************************************************************/
extern "C"
void Matrix_Transpose(const float *Matrix_in, int row, int col, float *Matrix_out)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float memory = 0;

	float* Matrix_din, *Matrix_dout;

	dim3 threads(MUL_BLOCK_SIZE, MUL_BLOCK_SIZE);
	int block_width = (row + MUL_BLOCK_SIZE -1)/MUL_BLOCK_SIZE;
	int block_height = (col+ MUL_BLOCK_SIZE -1)/MUL_BLOCK_SIZE;
	dim3 blocks(block_width,block_height);

	CUDA_SAFE_CALL(cudaMalloc((void**)&Matrix_dout,sizeof(float)*row*col));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Matrix_din,sizeof(float)*row*col));
	memory += 2 * sizeof(float)*row*col;

	CUDA_SAFE_CALL(cudaMemcpy(Matrix_din, Matrix_in, sizeof(float)*row*col,cudaMemcpyHostToDevice));

	Transpose_kernel<float><<<blocks, threads>>>(Matrix_din, row, col, Matrix_dout);
	cutilCheckMsg("kernel launch failure");

	CUDA_SAFE_CALL(cudaMemcpy(Matrix_out, Matrix_dout, sizeof(float)*row*col,cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(Matrix_dout));
	CUDA_SAFE_CALL(cudaFree(Matrix_din));


	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cout<<"函数运行时间:	"<<time<<" ms"<<std::endl;
	cout<<"函数使用显存:	"<<memory/1024<<" KB"<<endl;
}

/************************************************************************/
/*复数矩阵转置
/*@param Matrix_in :	输入 需要转置的矩阵 内存数据 一维连续存储
/*@param row			:	输入 矩阵行数
/*@param col			:	输入 矩阵列数
/*@param Matrix_out:	输出	转置后的矩阵
/************************************************************************/
extern "C"
void Matrix_TransposeCom(const float2 *Matrix_in, int row, int col, float2 *Matrix_out)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float memory = 0;

	float2* Matrix_din, *Matrix_dout;

	dim3 threads(MUL_BLOCK_SIZE, MUL_BLOCK_SIZE);
	int block_width = (row + MUL_BLOCK_SIZE -1)/MUL_BLOCK_SIZE;
	int block_height = (col+ MUL_BLOCK_SIZE -1)/MUL_BLOCK_SIZE;
	dim3 blocks(block_width,block_height);

	CUDA_SAFE_CALL(cudaMalloc((void**)&Matrix_dout,sizeof(float2)*row*col));
	CUDA_SAFE_CALL(cudaMalloc((void**)&Matrix_din,sizeof(float2)*row*col));
	memory += 2 * sizeof(float2)*row*col;

	CUDA_SAFE_CALL(cudaMemcpy(Matrix_din, Matrix_in, sizeof(float2)*row*col,cudaMemcpyHostToDevice));

	Transpose_kernel<float2><<<blocks, threads>>>(Matrix_din, row, col, Matrix_dout);
	cutilCheckMsg("kernel launch failure");

	CUDA_SAFE_CALL(cudaMemcpy(Matrix_out, Matrix_dout, sizeof(float2)*row*col,cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(Matrix_dout));
	CUDA_SAFE_CALL(cudaFree(Matrix_din));


	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cout<<"函数运行时间:	"<<time<<" ms"<<std::endl;
	cout<<"函数使用显存:	"<<memory/1024<<" KB"<<endl;
}