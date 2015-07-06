#ifndef __MATRIX_OPERATION_H__
#define __MATRIX_OPERATION_H__

/************************************************************************/
/*2012.3 Sheng Yang
/*矩阵基本操作 CPU端调用接口
/*首先工程应该导入文件Matrix_Operation.cu 和 Matrix_Operation.h
/*在需要使用到的文件中include "Matrix_Operation.h"
/*按照函数注释中的参数调用相应函数
/************************************************************************/


extern "C"
{
	/************************************************************************/
	/* 矩阵求最大值和最大值所在位置 flag = 0 按行求最大值 flag = 1 按列求最大值
	/* @param data_in	:	输入 矩阵数据 内存数据 一维连续存储
	/* @param row		:	输入 矩阵行数
	/* @param col		:	输入 矩阵列数
	/* @param index	:	输出 表示每行/每列最大值所在位置
	/* @param value	:	输出 表示每行/每列的最大值
	/* @param flag		:	输入 0 表示按行操作 1 表示按列操作
	/************************************************************************/
	void Matrix_Max(float *data_in, int row, int col, int *index, float *value, int flag);

	/************************************************************************/
	/*实数矩阵求和 flag = 0 按行求和 flag = 1 按列求和 flag = 2 全部求和
	/*@param data_in	:	输入 矩阵数据 内存数据 一维连续存储
	/*@param row		:	输入 矩阵行数
	/*@param col		:	输入 矩阵列数
	/*@param value	:	输出 每行/列求得的和
	/*@param flag		:	输入 0 按行操作 1 按列操作 2 全部数操作
	/************************************************************************/
	void Matrix_Sum(float *data_in, int row, int col, float *value, int flag);

	/************************************************************************/
	/*复数矩阵求和 flag = 0 按行求和 flag = 1 按列求和 flag = 2 全部求和
	/*@param data_in	:	输入 矩阵数据 内存数据 一维连续存储
	/*@param row		:	输入 矩阵行数
	/*@param col		:	输入 矩阵列数
	/*@param value	:	输出 每行/列求得的和
	/*@param flag		:	输入 0 按行操作 1 按列操作 2 全部数操作
	/************************************************************************/
	void Matrix_SumCom(cuComplex *data_in, int row, int col, cuComplex *value, int flag);

	/************************************************************************/
	/*实数矩阵乘法 
	/*@param Matrix_a	:	输入 矩阵A数据 内存数据 一维连续存储
	/*@param row_a		:	输入 矩阵A行数
	/*@param col_a		:	输入 矩阵A列数
	/*@param Matrix_b	:	输入 矩阵B数据 内存数据 一维连续存储
	/*@param row_b		:	输入 矩阵B行数
	/*@param col_b		:	输入 矩阵B列数
	/*@param Matrix_c	:	输出 矩阵C数据
	/*@param flag			:	输出 0 数据正常 1  相乘矩阵行列部匹配
	/************************************************************************/
	void Matrix_Multi(float *Matrix_a, int row_a, int col_a, float *Matrix_b, int row_b, int col_b, float *Matrix_c, int &flag);

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
	void Matrix_MultiCom(const float2 *Matrix_a, int row_a, int col_a, const float2 *Matrix_b, int row_b, int col_b, float2 *Matrix_c);

	/************************************************************************/
	/*实数矩阵转置
	/*@param Matrix_in :	输入 需要转置的矩阵 内存数据 一维连续存储
	/*@param row			:	输入 矩阵行数
	/*@param col			:	输入 矩阵列数
	/*@param Matrix_out:	输出	转置后的矩阵
	/************************************************************************/
	void Matrix_Transpose(const float *Matrix_in, int row, int col, float *Matrix_out);
	/************************************************************************/
	/*复数矩阵转置
	/*@param Matrix_in :	输入 需要转置的矩阵 内存数据 一维连续存储
	/*@param row			:	输入 矩阵行数
	/*@param col			:	输入 矩阵列数
	/*@param Matrix_out:	输出	转置后的矩阵
	/************************************************************************/
	void Matrix_TransposeCom(const float2 *Matrix_in, int row, int col, float2 *Matrix_out);


}

/***以下两个函数是为了满足 给定的需求 对函数进行了简单的封装和重载 (因为extern “C” 是C语言 不支持重载功能)***/

/************************************************************************/
/*复数矩阵求和 flag = 0 按行求和 flag = 1 按列求和 flag = 2 全部求和
/*@param data_in	:	输入 矩阵数据 内存数据 一维连续存储
/*@param row		:	输入 矩阵行数
/*@param col		:	输入 矩阵列数
/*@param value	:	输出 每行/列求得的和
/*@param flag		:	输入 0 按行操作 1 按列操作 2 全部数操作
/************************************************************************/
void Matrix_Sum(cuComplex *data_in, int row, int col, cuComplex *value, int flag)
{
	Matrix_SumCom(data_in, row, col, value, flag);
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
/*@param flag			:	输出 0 数据正常 1  相乘矩阵行列部匹配
/************************************************************************/
void Matrix_Multi(cuComplex * data_1, int row_1, int col_1, cuComplex *data_2, int row_2, int col_2, cuComplex *data_out, int &flag)
{
	if(col_1 != row_2)
	{
		cout<<"输入数据错误 操作矩阵行列数不匹配！"<<endl;
		return ;
	}
	Matrix_MultiCom(data_1,row_1,col_1,data_2,row_2,col_2,data_out);
}

/************************************************************************/
/*复数矩阵转置
/*@param Matrix_in :	输入 需要转置的矩阵 内存数据 一维连续存储
/*@param row			:	输入 矩阵行数
/*@param col			:	输入 矩阵列数
/*@param Matrix_out:	输出	转置后的矩阵
/************************************************************************/
void Matrix_Transpose(const cuComplex *Matrix_in, int row, int col, cuComplex *Matrix_out)
{
	Matrix_TransposeCom(Matrix_in,row,col,Matrix_out);
}

#endif