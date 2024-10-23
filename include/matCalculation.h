#pragma once

#include <vulkan/vulkan.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>

class Matrix {
private:
	vk::UniqueInstance m_instance;
	vk::PhysicalDevice m_physicalDevice;
	vk::UniqueDevice m_device;
	vk::Queue m_queue;
	vk::UniqueCommandPool m_commandPool;
	vk::UniqueSemaphore m_semaphore;
	vk::UniqueFence m_fence;

	// vulkanの初期化をする関数. 各変数を決める.
	void initVulkan();

	// バッファ作成で適したメモリタイプをとってくる関数.
	uint32_t findMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags property);

	// バッファを作成する関数.
	vk::UniqueBuffer createBuffer(const vk::Device device, const vk::PhysicalDevice physicalDevice, const vk::DeviceSize bufferSize,
		const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags property, vk::UniqueDeviceMemory& bufferMemory);

	// srcBufferからdstBufferへデータをコピーするコマンドバッファを作成する関数.
	vk::UniqueCommandBuffer makeCopyCommandBuffer(const vk::Device device, vk::CommandPool commandPool, vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
		vk::DeviceSize bufferSize, vk::CommandBufferLevel level);

	// spvファイルを読み込む関数.
	std::vector<char> readFile(const std::string& filename);

	// 読み込んだspvファイルからシェーダーを作る関数.
	vk::UniqueShaderModule createShaderModule(const vk::Device device, const std::vector<char>& code);

public:
	Matrix();
	~Matrix();

	std::vector<std::vector<float>> multi(const std::vector<std::vector<float>> &matA, const std::vector<std::vector<float>> &matB);

	std::vector<std::vector<float>> multiEach(const std::vector<std::vector<float>> &matA, const std::vector<std::vector<float>> &matB);

	std::vector<std::vector<float>> sum(const std::vector<std::vector<float>> &matA, const std::vector<std::vector<float>> &matB);

	std::vector<std::vector<float>> sum(const std::vector<std::vector<float>>& matA, const float value);

	std::vector<std::vector<float>> diff(const std::vector<std::vector<float>> &matA, const std::vector<std::vector<float>> &matB);
};