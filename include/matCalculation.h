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

	// vulkan�̏�����������֐�. �e�ϐ������߂�.
	void initVulkan();

	// �o�b�t�@�쐬�œK�����������^�C�v���Ƃ��Ă���֐�.
	uint32_t findMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags property);

	// �o�b�t�@���쐬����֐�.
	vk::UniqueBuffer createBuffer(const vk::Device device, const vk::PhysicalDevice physicalDevice, const vk::DeviceSize bufferSize,
		const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags property, vk::UniqueDeviceMemory& bufferMemory);

	// srcBuffer����dstBuffer�փf�[�^���R�s�[����R�}���h�o�b�t�@���쐬����֐�.
	vk::UniqueCommandBuffer makeCopyCommandBuffer(const vk::Device device, vk::CommandPool commandPool, vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
		vk::DeviceSize bufferSize, vk::CommandBufferLevel level);

	// spv�t�@�C����ǂݍ��ފ֐�.
	std::vector<char> readFile(const std::string& filename);

	// �ǂݍ���spv�t�@�C������V�F�[�_�[�����֐�.
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