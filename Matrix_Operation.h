#ifndef __MATRIX_OPERATION_H__
#define __MATRIX_OPERATION_H__

/************************************************************************/
/*2012.3 Sheng Yang
/*����������� CPU�˵��ýӿ�
/*���ȹ���Ӧ�õ����ļ�Matrix_Operation.cu �� Matrix_Operation.h
/*����Ҫʹ�õ����ļ���include "Matrix_Operation.h"
/*���պ���ע���еĲ���������Ӧ����
/************************************************************************/


extern "C"
{
	/************************************************************************/
	/* ���������ֵ�����ֵ����λ�� flag = 0 ���������ֵ flag = 1 ���������ֵ
	/* @param data_in	:	���� �������� �ڴ����� һά�����洢
	/* @param row		:	���� ��������
	/* @param col		:	���� ��������
	/* @param index	:	��� ��ʾÿ��/ÿ�����ֵ����λ��
	/* @param value	:	��� ��ʾÿ��/ÿ�е����ֵ
	/* @param flag		:	���� 0 ��ʾ���в��� 1 ��ʾ���в���
	/************************************************************************/
	void Matrix_Max(float *data_in, int row, int col, int *index, float *value, int flag);

	/************************************************************************/
	/*ʵ��������� flag = 0 ������� flag = 1 ������� flag = 2 ȫ�����
	/*@param data_in	:	���� �������� �ڴ����� һά�����洢
	/*@param row		:	���� ��������
	/*@param col		:	���� ��������
	/*@param value	:	��� ÿ��/����õĺ�
	/*@param flag		:	���� 0 ���в��� 1 ���в��� 2 ȫ��������
	/************************************************************************/
	void Matrix_Sum(float *data_in, int row, int col, float *value, int flag);

	/************************************************************************/
	/*����������� flag = 0 ������� flag = 1 ������� flag = 2 ȫ�����
	/*@param data_in	:	���� �������� �ڴ����� һά�����洢
	/*@param row		:	���� ��������
	/*@param col		:	���� ��������
	/*@param value	:	��� ÿ��/����õĺ�
	/*@param flag		:	���� 0 ���в��� 1 ���в��� 2 ȫ��������
	/************************************************************************/
	void Matrix_SumCom(cuComplex *data_in, int row, int col, cuComplex *value, int flag);

	/************************************************************************/
	/*ʵ������˷� 
	/*@param Matrix_a	:	���� ����A���� �ڴ����� һά�����洢
	/*@param row_a		:	���� ����A����
	/*@param col_a		:	���� ����A����
	/*@param Matrix_b	:	���� ����B���� �ڴ����� һά�����洢
	/*@param row_b		:	���� ����B����
	/*@param col_b		:	���� ����B����
	/*@param Matrix_c	:	��� ����C����
	/*@param flag			:	��� 0 �������� 1  ��˾������в�ƥ��
	/************************************************************************/
	void Matrix_Multi(float *Matrix_a, int row_a, int col_a, float *Matrix_b, int row_b, int col_b, float *Matrix_c, int &flag);

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
	void Matrix_MultiCom(const float2 *Matrix_a, int row_a, int col_a, const float2 *Matrix_b, int row_b, int col_b, float2 *Matrix_c);

	/************************************************************************/
	/*ʵ������ת��
	/*@param Matrix_in :	���� ��Ҫת�õľ��� �ڴ����� һά�����洢
	/*@param row			:	���� ��������
	/*@param col			:	���� ��������
	/*@param Matrix_out:	���	ת�ú�ľ���
	/************************************************************************/
	void Matrix_Transpose(const float *Matrix_in, int row, int col, float *Matrix_out);
	/************************************************************************/
	/*��������ת��
	/*@param Matrix_in :	���� ��Ҫת�õľ��� �ڴ����� һά�����洢
	/*@param row			:	���� ��������
	/*@param col			:	���� ��������
	/*@param Matrix_out:	���	ת�ú�ľ���
	/************************************************************************/
	void Matrix_TransposeCom(const float2 *Matrix_in, int row, int col, float2 *Matrix_out);


}

/***��������������Ϊ������ ���������� �Ժ��������˼򵥵ķ�װ������ (��Ϊextern ��C�� ��C���� ��֧�����ع���)***/

/************************************************************************/
/*����������� flag = 0 ������� flag = 1 ������� flag = 2 ȫ�����
/*@param data_in	:	���� �������� �ڴ����� һά�����洢
/*@param row		:	���� ��������
/*@param col		:	���� ��������
/*@param value	:	��� ÿ��/����õĺ�
/*@param flag		:	���� 0 ���в��� 1 ���в��� 2 ȫ��������
/************************************************************************/
void Matrix_Sum(cuComplex *data_in, int row, int col, cuComplex *value, int flag)
{
	Matrix_SumCom(data_in, row, col, value, flag);
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
/*@param flag			:	��� 0 �������� 1  ��˾������в�ƥ��
/************************************************************************/
void Matrix_Multi(cuComplex * data_1, int row_1, int col_1, cuComplex *data_2, int row_2, int col_2, cuComplex *data_out, int &flag)
{
	if(col_1 != row_2)
	{
		cout<<"�������ݴ��� ����������������ƥ�䣡"<<endl;
		return ;
	}
	Matrix_MultiCom(data_1,row_1,col_1,data_2,row_2,col_2,data_out);
}

/************************************************************************/
/*��������ת��
/*@param Matrix_in :	���� ��Ҫת�õľ��� �ڴ����� һά�����洢
/*@param row			:	���� ��������
/*@param col			:	���� ��������
/*@param Matrix_out:	���	ת�ú�ľ���
/************************************************************************/
void Matrix_Transpose(const cuComplex *Matrix_in, int row, int col, cuComplex *Matrix_out)
{
	Matrix_TransposeCom(Matrix_in,row,col,Matrix_out);
}

#endif