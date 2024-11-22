#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include<chrono>
#include<thread>
#include<fstream>
#include "matCalculation.h"


const int WIDTHA = 1000;
const int HEIGHTA = 1000;
const int WIDTHB = 1000;
const int HEIGHTB = 1000;

std::vector<std::vector<float>> multiplyMatrix(
	const std::vector<std::vector<float>>& A,
	const std::vector<std::vector<float>>& B) {
	int rowsA = (int)A.size();
	int colsA = (int)A[0].size();
	int colsB = (int)B[0].size();

	std::vector<std::vector<float>> result(rowsA, std::vector<float>(colsB, 0.0f));

	for (int i = 0; i < rowsA; ++i) {
		for (int j = 0; j < colsB; ++j) {
			for (int k = 0; k < colsA; ++k) {
				result[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return result;
}

int main() {
	//std::vector<std::vector<std::complex<float>>> hoge = { {{1.0f, 1.0f}, {2.0f, 1.0f}, {3.0f, 1.0f} }, { {0.0f, 1.0f}, {0.0f, 2.0f}, {0.0f, 3.0f} }, { {0.0f, 3.0f}, {0.0f, 2.0f}, {0.0f, 1.0f} }};
	//std::vector<std::vector<std::complex<float>>> fuga = { {{0.0f, 2.0f}, {1.0f, 0.0f}, {0.0f, 3.0f}}, {{1.0f, 2.0f}, {2.0f, 3.0f}, {4.0f, 3.0f}}, {{0.0f, 1.0f}, {0.0f, 3.0f}, {0.0f, 2.0f}} };

	//Matrix calcInstance;
	//std::vector<std::vector<std::complex<float>>> ans = calcInstance.complexSum(hoge, fuga);

	//for (int i = 0; i < 3; ++i) {
	//	for (int j = 0; j < 3; ++j) {
	//		std::cout << "row: " << i << " col: " << j << " value: " << ans[i][j].real() <<
	//			"+" << ans[i][j].imag() << "i" << std::endl;
	//	}
	//}


	int size[7] = { 10, 50, 200, 400, 800, 1000, 1500 };

	std::ofstream outputFile("gpu_calculation.txt");

	if (!outputFile.is_open()) {
		std::cerr << "Error: Could not open file for writing." << std::endl;
		return 1;
	}

	int count = 5;

	for (int &s : size) {
		std::vector<std::vector<float>> A(s, std::vector<float>(s, 2.5f));
		std::vector<std::vector<float>> B(s, std::vector<float>(s, 4.2f));
		Matrix calcInstance;

		double time = 0;

		for (int j = 0; j < count; ++j) {
			Matrix calcInstance;
			auto start = std::chrono::high_resolution_clock::now();
			std::vector<std::vector<float>> C = calcInstance.multi(A, B);
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> t = end - start;
			time += t.count();
			std::cout << s << "~" << s << "“¯Žm‚Ìs—ñÏ‚Ì‰‰ŽZŽžŠÔ: " << t.count() << " s" << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}

		outputFile << s << ", " << time / count << std::endl;
	}

	outputFile.close();

}
