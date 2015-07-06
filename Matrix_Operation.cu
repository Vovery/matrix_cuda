/************************************************************************/
/*2012-3 Sheng Yang
/*����������� CUDA ���� �˺���ʵ�� ���߳̿黮�ֵ�
/************************************************************************/

#include <cuda_runtime.h>
#include <cutil.h>
#include <iostream>
#include <cutil_inline.h>
#include <cufft.h>

using namespace std;


#define MIN_DATA -999999999 //�����ֵʱ�õ�����Сֵ
#define BLOCK_SIZE 512		//һά���ֿ��С
#define MUL_BLOCK_SIZE 16	//��ά���ֿ��С
#define KAHAN_SUMMATION_FORMULA 0 //�Ƿ����ÿ��๫ʽ������͹����е�����ۻ�

//�����������ֵ
//����˼��: ÿһ��Block����һ�� ����������ϲ�����
//���水�в��������� Ҳ��ͬһ˼�� ����׸��
//����shared memory���в��й�Լʽ�ӷ���
/************************************************************************/
/* A-------A------A-------A
/*   \   /           \  /
/*     2A             2A
/*       \           /
/*	           4A
/************************************************************************/
//�����ᵽ��Լ�㷨��������˼��
//@param data_in	���� ����
//@param row		���� ������
//@param column		���� ������
//@param index		��� ���ֵ����λ��
//@param value		��� ���ֵ
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
//�����������ֵ
//����˼�룺Ϊ������ϲ����ʵ�Ҫ�� ÿ���鸺����128(��ʵ���Ը�С)�е�����
//���水�е��㷨��ֻ��BLOCK_SIZE = 512���� �ı���BLOCK_SZIE��С�Ļ� Ҫ���ĺ˺���
//���ÿһ��128�� ���߳�ID = ida 
//���߳�IDΪ ida%,(ida+128)%512,(ida+256)%512 (ida+384)%512 ����ͬһ�е���������
//������һ����Ϳ������128�е�ÿ�е����ֵ ���°������Ҳ��������˼�� ����׸��
//@param data_in	���� ����
//@param row		���� ������
//@param column		���� ������
//@param index		��� ���ֵ����λ��
//@param value		��� ���ֵ
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

//ʵ���������
//@param data_in	���� ����
//@param row		���� ������
//@param column		���� ������
//@param value		��� ��
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

//ʵ���������
//@param data_in	���� ����
//@param row		���� ������
//@param column		���� ������
//@param value		��� ��
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

//ʵ�������ܺ� 
//������ʵ�ȶԾ�������˰�����Ͳ��� 
//���ʵ�ֵ���һ�����һλ���ݵĹ�Լ��Ͳ���
//@param temp_value		��������
//@param num			���ݸ���
//@param value			�����	
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

//�����������
//@param data_in	���� ����
//@param row		���� ������
//@param column		���� ������
//@param value		��� ��
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

//�����������
//@param data_in	���� ����
//@param row		���� ������
//@param column		���� ������
//@param value		��� ��
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

//�������ܺ�
//@param data_in	���� ����
//@param row		���� ������
//@param column		���� ������
//@param value		��� ��
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

//ʵ������˷�
//����˼�룺ÿ���鸺��һ�����������16*16С��Ľ������ 
//����shared memory��ʵ�ֶ�global memory�ĺϲ����ʺʹ洢
//@param Matrix_a	���� ����A
//@param row_a		���� ����A��
//@param col_a		���� ����A��
//@param lda		���� ����A���Դ���ʵ�ʵ�ÿ�д�С
//@param Matrix_b	���� ����B
//@param row_b		���� ����B��
//@param col_b		���� ����B��
//@param ldb		���� ����B���Դ���ʵ�ʵ�ÿ�д�С
//@param Matrix_c	��� ���� A*B
//@param ldc		���� ����C���Դ���ʵ�ʵ�ÿ�д�С
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

//��������˷�
//����˼�룺ÿ���鸺��һ�����������16*16С��Ľ������ 
//����shared memory��ʵ�ֶ�global memory�ĺϲ����ʺʹ洢
//@param Matrix_a	���� ����A
//@param row_a		���� ����A��
//@param col_a		���� ����A��
//@param lda		���� ����A���Դ���ʵ�ʵ�ÿ�д�С
//@param Matrix_b	���� ����B
//@param row_b		���� ����B��
//@param col_b		���� ����B��
//@param ldb		���� ����B���Դ���ʵ�ʵ�ÿ�д�С
//@param Matrix_c	��� ���� A*B
//@param ldc		���� ����C���Դ���ʵ�ʵ�ÿ�д�С
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

//����ת��
//����˼��:����sharedmemoryʵ��С��16*16��ת�� ʵ�ֶ�ȡд��ĺϲ�
//@param T					���� ������������
//@param Matrix_in		���� ��������
//@param row				����	��������
//@param col				���� ��������
//@param Matrix_out	���	ת�ú�ľ���
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
/* ���������ֵ�����ֵ����λ�� flag = 0 ���������ֵ flag = 1 ���������ֵ
/* @param data_in	:	���� �������� �ڴ����� һά�����洢
/* @param row		:	���� ��������
/* @param col		:	���� ��������
/* @param index		:	��� ��ʾÿ��/ÿ�����ֵ����λ��
/* @param value		:	��� ��ʾÿ��/ÿ�е����ֵ
/* @param flag		:	���� 0 ��ʾ���в��� 1 ��ʾ���в���
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

	cout<<"��������ʱ��:	"<<time<<" ms"<<std::endl;
	cout<<"����ʹ���Դ�:	"<<memory/1024<<" KB"<<endl;
}

/************************************************************************/
/*ʵ��������� flag = 0 ������� flag = 1 ������� flag = 2 ȫ�����
/*@param data_in	:	���� �������� �ڴ����� һά�����洢
/*@param row		:	���� ��������
/*@param col		:	���� ��������
/*@param value		:	��� ÿ��/����õĺ�
/*@param flag		:	���� 0 ���в��� 1 ���в��� 2 ȫ��������
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

	cout<<"��������ʱ��:	"<<time<<" ms"<<std::endl;
	cout<<"����ʹ���Դ�:	"<<memory/1024<<" KB"<<endl;

}

/************************************************************************/
/*����������� flag = 0 ������� flag = 1 ������� flag = 2 ȫ�����
/*@param data_in	:	���� �������� �ڴ����� һά�����洢
/*@param row		:	���� ��������
/*@param col		:	���� ��������
/*@param value		:	��� ÿ��/����õĺ�
/*@param flag		:	���� 0 ���в��� 1 ���в��� 2 ȫ��������
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

	cout<<"��������ʱ��:	"<<time<<" ms"<<std::endl;
	cout<<"����ʹ���Դ�:	"<<memory/1024<<" KB"<<endl;
}

/************************************************************************/
/*ʵ������˷� 
/*@param Matrix_a	:	���� ����A���� �ڴ����� һά�����洢
/*@param row_a		:	���� ����A����
/*@param col_a		:	���� ����A����
/*@param Matrix_b	:	���� ����B���� �ڴ����� һά�����洢
/*@param row_b		:	���� ����B����
/*@param col_b		:	���� ����B����
/*@param Matrix_c	:	��� ����C����
/*@param flag		:	��� 0 �������� 1  ��˾������в�ƥ��
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

	//�����Դ���� Ϊ�˷���16*16��ĳ˷����� ʹ��cudaMallocPitch�ƶ�����
	//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
	//dst ����Ŀ���ַ src ����Դ��ַ
	//dpitch ��ָ ���Ƶ�Ŀ���ַ��ʵ�ʵ�ÿ�����ݴ�С(byte)
	//spitch ��ָ ��Դ��ַ�� ���Ƶ���Ӧ�е�ʵ�ʵ�ַ��С(byte)
	//width  ��ָ ���Ƶ�Ŀ���ַ��ÿ�еĿ��(byte)
	//height ��ָ ���Ƶ�Ŀ���ַ������(��)
	//kind	 ��ָ ����������

	size_t pitch_a, pitch_b, pitch_c;

	//Ϊ�˶����ʹ������16�ı��� ����������Դ����

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

	//������������������
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
	cout<<"��������ʱ��:	"<<time<<" ms"<<std::endl;
	cout<<"����ʹ���Դ�:	"<<memory/1024<<" KB"<<endl;
}

/************************************************************************/
/*��������˷� 
/*@param Matrix_a	:	���� ����A���� �ڴ����� һά�����洢
/*@param row_a		:	���� ����A����
/*@param col_a		:	���� ����A����
/*@param Matrix_b	:	���� ����B���� �ڴ����� һά�����洢
/*@param row_b		:	���� ����B����
/*@param col_b		:	���� ����B����
/*@param Matrix_c	:	��� ����C����
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

	//�����Դ���� Ϊ�˷���16*16��ĳ˷����� ʹ��cudaMallocPitch�ƶ�����

	//cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
	//dst ����Ŀ���ַ src ����Դ��ַ
	//dpitch ��ָ ���Ƶ�Ŀ���ַ��ʵ�ʵ�ÿ�����ݴ�С(byte)
	//spitch ��ָ ��Դ��ַ�� ���Ƶ���Ӧ�е�ʵ�ʵ�ַ��С(byte)
	//width  ��ָ ���Ƶ�Ŀ���ַ��ÿ�еĿ��(byte)
	//height ��ָ ���Ƶ�Ŀ���ַ������(��)
	//kind	 ��ָ ����������

	size_t pitch_a, pitch_b, pitch_c;

	//Ϊ�˶����ʹ������16�ı��� ����������Դ����

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
	cout<<"��������ʱ��:	"<<time<<" ms"<<std::endl;
	cout<<"����ʹ���Դ�:	"<<memory/1024<<" KB"<<endl;
}

/************************************************************************/
/*ʵ������ת��
/*@param Matrix_in :	���� ��Ҫת�õľ��� �ڴ����� һά�����洢
/*@param row			:	���� ��������
/*@param col			:	���� ��������
/*@param Matrix_out:	���	ת�ú�ľ���
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
	cout<<"��������ʱ��:	"<<time<<" ms"<<std::endl;
	cout<<"����ʹ���Դ�:	"<<memory/1024<<" KB"<<endl;
}

/************************************************************************/
/*��������ת��
/*@param Matrix_in :	���� ��Ҫת�õľ��� �ڴ����� һά�����洢
/*@param row			:	���� ��������
/*@param col			:	���� ��������
/*@param Matrix_out:	���	ת�ú�ľ���
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
	cout<<"��������ʱ��:	"<<time<<" ms"<<std::endl;
	cout<<"����ʹ���Դ�:	"<<memory/1024<<" KB"<<endl;
}