#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include "matCalculation.h"


const int WIDTHA = 1000;
const int HEIGHTA = 1000;
const int WIDTHB = 1000;
const int HEIGHTB = 1000;

class Mat {
private:
	std::vector<std::vector<float>> mat;
	vk::UniqueInstance instance;
	vk::PhysicalDevice physicalDevice;
	vk::UniqueDevice device;
	vk::Queue queue;

public:
	Mat() {}
	Mat(std::vector<std::vector<float>> v) : mat(v) {}
	//initVulkan();
};


// 適したメモリタイプの場所を返す関数.
uint32_t findMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags property) {
	vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if (memoryProperties.memoryTypes[i].propertyFlags & property) {
			if (typeFilter & (1 << i)) {
				return i;
			}
		}
	}

	throw std::runtime_error("failedt to findMemoryType.");
}

// バッファを作る関数.
vk::UniqueBuffer createBuffer(const vk::Device device, const vk::PhysicalDevice physicalDevice, const vk::DeviceSize bufferSize,
	const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags property, vk::UniqueDeviceMemory& bufferMemory)
{
	vk::BufferCreateInfo bufferInfo = {};
	bufferInfo.size = bufferSize;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = vk::SharingMode::eExclusive; // 同時に複数のキューから操作しない.

	vk::UniqueBuffer buffer = device.createBufferUnique(bufferInfo);

	vk::MemoryRequirements memReq = device.getBufferMemoryRequirements(buffer.get());
	vk::MemoryAllocateInfo memInfo = {};
	memInfo.allocationSize = memReq.size;
	memInfo.memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits, property);

	bufferMemory = device.allocateMemoryUnique(memInfo);
	device.bindBufferMemory(buffer.get(), bufferMemory.get(), 0);

	return std::move(buffer);
}

vk::UniqueCommandBuffer makeCopyCommandBuffer(const vk::Device device, vk::CommandPool commandPool, vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
	vk::DeviceSize bufferSize, vk::CommandBufferLevel level)
{
	vk::CommandBufferAllocateInfo allocInfo = {};
	allocInfo.commandPool = commandPool;
	allocInfo.level = level;
	allocInfo.commandBufferCount = 1;

	vk::UniqueCommandBuffer commandBuffer = std::move(device.allocateCommandBuffersUnique(allocInfo)[0]);

	vk::CommandBufferBeginInfo beginInfo = {};
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	commandBuffer->begin(beginInfo);

	vk::BufferCopy copyRegion = {};
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = bufferSize;

	commandBuffer->copyBuffer(srcBuffer, dstBuffer, copyRegion);
	commandBuffer->end();

	return std::move(commandBuffer);
}

// spvファイルを読み込む関数.
std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open spv file.");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

// 読み込んだspvファイルからシェーダーを作る関数.
vk::UniqueShaderModule createShaderModule(const vk::Device device, const std::vector<char>& code) {
	vk::ShaderModuleCreateInfo shaderInfo = {};
	shaderInfo.codeSize = code.size();
	shaderInfo.pCode = reinterpret_cast<const uint32_t*> (code.data());
	return std::move(device.createShaderModuleUnique(shaderInfo));
}




int main() {
	//std::vector<std::vector<float>> hoge{ {1.0f, 2.0f, 3.0f}, {4.0f, 7.0f, 6.0f}, {7.0f, 4.0f, 9.0f} };
	//std::vector<std::vector<float>> fuga{ {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f} };
	//std::vector<std::vector<float>> hoge(1000, std::vector<float>(1000, 2.0f));
	//std::vector<std::vector<float>> fuga(1000, std::vector<float>(1000, 4.2f));
	std::vector<std::vector<std::complex<float>>> hoge = { {{1.0f, 1.0f}, {2.0f, 1.0f}, {3.0f, 1.0f} }, { {0.0f, 1.0f}, {0.0f, 2.0f}, {0.0f, 3.0f} }, { {0.0f, 3.0f}, {0.0f, 2.0f}, {0.0f, 1.0f} }};
	std::vector<std::vector<std::complex<float>>> fuga = { {{0.0f, 2.0f}, {1.0f, 0.0f}, {0.0f, 3.0f}}, {{1.0f, 2.0f}, {2.0f, 3.0f}, {4.0f, 3.0f}}, {{0.0f, 1.0f}, {0.0f, 3.0f}, {0.0f, 2.0f}} };

	Matrix calcInstance;
	//std::vector<std::vector<float>> ans = calcInstance.multiple(hoge, fuga);
	std::vector<std::vector<std::complex<float>>> ans = calcInstance.complexMulti(hoge, fuga);

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			std::cout << "row: " << i << " col: " << j << " value: " << ans[i][j].real() <<
				"+" << ans[i][j].imag() << "i" << std::endl;
		}
	}
}
