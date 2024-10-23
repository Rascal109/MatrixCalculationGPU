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
	std::vector<std::vector<float>> hoge{ {1.0f, 2.0f, 3.0f}, {4.0f, 7.0f, 6.0f}, {7.0f, 4.0f, 9.0f} };
	std::vector<std::vector<float>> fuga{ {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f} };
	//std::vector<std::vector<float>> hoge(1000, std::vector<float>(1000, 2.0f));
	//std::vector<std::vector<float>> fuga(1000, std::vector<float>(1000, 4.2f));

	Matrix calcInstance;
	//std::vector<std::vector<float>> ans = calcInstance.multiple(hoge, fuga);
	std::vector<std::vector<float>> ans = calcInstance.multiEach(hoge, fuga);

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			std::cout << "row: " << i << " col: " << j << " value: " << ans[i][j] << std::endl;
		}
	}
}
//
//int main() {
//	vk::ApplicationInfo appInfo = {};
//	appInfo.pApplicationName = "MatrixCalculation";
//	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
//	appInfo.pEngineName = "No Engine";
//	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
//	appInfo.apiVersion = VK_API_VERSION_1_0;
//
//	vk::InstanceCreateInfo instanceInfo = {};
//
//	// バリデーションレイヤーの使用.
//	const std::vector<const char*> layers = {
//		"VK_LAYER_KHRONOS_validation"
//	};
//	instanceInfo.pApplicationInfo = &appInfo;
//	instanceInfo.ppEnabledLayerNames = layers.data();
//	instanceInfo.enabledLayerCount = static_cast<uint32_t> (layers.size());
//
//	vk::UniqueInstance instance = vk::createInstanceUnique(instanceInfo);
//
//	vk::PhysicalDevice physicalDevice = instance->enumeratePhysicalDevices()[0];
//
//	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
//	uint32_t queueFamilyIndex = -1;
//	for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
//		if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
//			queueFamilyIndex = i;
//			break;
//		}
//	}
//
//	if (queueFamilyIndex == -1) {
//		throw std::runtime_error("failed to find queueFamilyIndex.");
//	}
//
//	// 論理デバイスとキューの作成.
//	float queuePriority = 0.0f;
//	vk::DeviceQueueCreateInfo queueCreateInfo = {};
//	queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
//	queueCreateInfo.queueCount = 1;
//	queueCreateInfo.pQueuePriorities = &queuePriority;
//
//	std::vector<vk::DeviceQueueCreateInfo> queues{ queueCreateInfo };
//
//	vk::DeviceCreateInfo deviceCreateInfo = {};
//	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t> (queues.size());
//	deviceCreateInfo.pQueueCreateInfos = queues.data();
//	deviceCreateInfo.enabledLayerCount = static_cast<uint32_t> (layers.size());
//	deviceCreateInfo.ppEnabledLayerNames = layers.data();
//
//	vk::UniqueDevice device = physicalDevice.createDeviceUnique(deviceCreateInfo);
//	vk::Queue queue = device->getQueue(queueFamilyIndex, 0);
//
//	// セマフォとフェンスの作成.
//	vk::UniqueSemaphore semaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
//	vk::UniqueFence fence = device->createFenceUnique({});
//
//	// コマンドプールの作成.
//	vk::CommandPoolCreateInfo poolCreateInfo = {};
//	poolCreateInfo.queueFamilyIndex = queueFamilyIndex;
//	vk::UniqueCommandPool commandPool = device->createCommandPoolUnique(poolCreateInfo);
//
//	// スレッドのサイズの表示.
//	vk::PhysicalDeviceProperties physicalProp = physicalDevice.getProperties();
//
//	std::cout << "max compute work group num: "
//		<< physicalProp.limits.maxComputeWorkGroupCount[0] << ", "
//		<< physicalProp.limits.maxComputeWorkGroupCount[1] << ", "
//		<< physicalProp.limits.maxComputeWorkGroupCount[2] << std::endl;
//
//	std::cout << "max compute work group size: "
//		<< physicalProp.limits.maxComputeWorkGroupSize[0] << ", "
//		<< physicalProp.limits.maxComputeWorkGroupSize[1] << ", "
//		<< physicalProp.limits.maxComputeWorkGroupSize[2] << std::endl;
//
//	std::cout << "max thread size: "
//		<< physicalProp.limits.maxComputeWorkGroupInvocations << std::endl;
//
//	// バッファの作成.
//	std::vector<std::vector<float>> matrixA(HEIGHTA, std::vector<float>(WIDTHA, 1.0f));
//	std::vector<std::vector<float>> matrixB(HEIGHTB, std::vector<float>(WIDTHB, 2.0f));
//	std::vector<std::vector<float>> matrixC(HEIGHTA, std::vector<float>(WIDTHB));
//
//	std::vector<float> dataA(WIDTHA * HEIGHTA);
//	std::vector<float> dataB(WIDTHB * HEIGHTB);
//	std::vector<float> dataC(WIDTHB * HEIGHTA);
//
//	for (int i = 0; i < HEIGHTA; ++i) {
//		for (int j = 0; j < WIDTHA; ++j) {
//			dataA[i * WIDTHA + j] = matrixA[i][j];
//		}
//	}
//	for (int i = 0; i < HEIGHTB; ++i) {
//		for (int j = 0; j < WIDTHB; ++j) {
//			dataB[i * WIDTHB + j] = matrixB[i][j];
//		}
//	}
//
//	vk::DeviceSize bufferSizeInA = sizeof(float) * WIDTHA * HEIGHTA;
//	vk::DeviceSize bufferSizeInB = sizeof(float) * WIDTHB * HEIGHTB;
//	vk::DeviceSize bufferSizeOut = sizeof(float) * HEIGHTA * WIDTHB;
//
//	vk::UniqueDeviceMemory stagingBufferMemoryA, stagingBufferMemoryB, stagingBufferMemoryC, bufferAMemory, bufferBMemory, bufferCMemory;
//
//	vk::UniqueBuffer stagingBufferA = createBuffer(device.get(), physicalDevice, bufferSizeInA,
//		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
//		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryA);	
//
//	vk::UniqueBuffer stagingBufferB = createBuffer(device.get(), physicalDevice, bufferSizeInB,
//		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
//		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryB);	
//	
//	vk::UniqueBuffer stagingBufferC = createBuffer(device.get(), physicalDevice, bufferSizeOut,
//		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
//		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryC);
//
//	vk::UniqueBuffer bufferA = createBuffer(device.get(), physicalDevice, bufferSizeInA,
//		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
//		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferAMemory);
//
//	vk::UniqueBuffer bufferB = createBuffer(device.get(), physicalDevice, bufferSizeInB,
//		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
//		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferBMemory);
//
//	vk::UniqueBuffer bufferC = createBuffer(device.get(), physicalDevice, bufferSizeOut,
//		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
//		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferCMemory);
//
//	// コマンドバッファの作成.
//	void* data;
//	data = device->mapMemory(stagingBufferMemoryA.get(), 0, bufferSizeInA);
//	std::memcpy(data, dataA.data(), bufferSizeInA);
//	vk::UniqueCommandBuffer commandBufferA = makeCopyCommandBuffer(device.get(), commandPool.get(),
//		stagingBufferA.get(), bufferA.get(), bufferSizeInA, vk::CommandBufferLevel::ePrimary);
//	device->unmapMemory(stagingBufferMemoryA.get());
//
//	data = device->mapMemory(stagingBufferMemoryB.get(), 0, bufferSizeInB);
//	std::memcpy(data, dataB.data(), bufferSizeInB);
//	vk::UniqueCommandBuffer commandBufferB = makeCopyCommandBuffer(device.get(), commandPool.get(),
//		stagingBufferB.get(), bufferB.get(), bufferSizeInB, vk::CommandBufferLevel::ePrimary);
//	device->unmapMemory(stagingBufferMemoryB.get());
//
//	std::vector<vk::CommandBuffer> commandBuffers = { commandBufferA.get(), commandBufferB.get() };
//
//	// キューに流して実行.
//	vk::SubmitInfo submitInfo = {};
//	submitInfo.commandBufferCount = static_cast<uint32_t> (commandBuffers.size());
//	submitInfo.pCommandBuffers = commandBuffers.data();
//	submitInfo.signalSemaphoreCount = 1;
//	submitInfo.setPSignalSemaphores(&semaphore.get());
//	queue.submit(submitInfo);
//
//	// シェーダモジュールの設定. 修正必要.
//	std::vector<char> code = readFile("matMultiple.spv");
//	vk::UniqueShaderModule computeShaderModule = createShaderModule(device.get(), code);
//
//	// デスクリプタプールの作成.
//	vk::DescriptorPoolSize descPoolSize = {};
//	descPoolSize.type = vk::DescriptorType::eStorageBuffer;
//	descPoolSize.descriptorCount = 3; // デスクリプタに結び付けるバッファの数.
//
//	vk::DescriptorPoolCreateInfo descPoolInfo = {};
//	descPoolInfo.poolSizeCount = 1;
//	descPoolInfo.pPoolSizes = &descPoolSize;
//	descPoolInfo.maxSets = 1;
//	descPoolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
//	vk::UniqueDescriptorPool descPool = device->createDescriptorPoolUnique(descPoolInfo);
//
//	// デスクリプタセットレイアウトの作成.
//	std::vector<vk::DescriptorSetLayoutBinding> descSetLayoutBinding(3);
//	descSetLayoutBinding[0].binding = 0;
//	descSetLayoutBinding[0].descriptorType = vk::DescriptorType::eStorageBuffer;
//	descSetLayoutBinding[0].descriptorCount = 1;
//	descSetLayoutBinding[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
//
//	descSetLayoutBinding[1].binding = 1;
//	descSetLayoutBinding[1].descriptorType = vk::DescriptorType::eStorageBuffer;
//	descSetLayoutBinding[1].descriptorCount = 1;
//	descSetLayoutBinding[1].stageFlags = vk::ShaderStageFlagBits::eCompute;
//
//	descSetLayoutBinding[2].binding = 2;
//	descSetLayoutBinding[2].descriptorType = vk::DescriptorType::eStorageBuffer;
//	descSetLayoutBinding[2].descriptorCount = 1;
//	descSetLayoutBinding[2].stageFlags = vk::ShaderStageFlagBits::eCompute;
//
//	vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo = {};
//	descSetLayoutInfo.bindingCount = static_cast<uint32_t> (descSetLayoutBinding.size());
//	descSetLayoutInfo.pBindings = descSetLayoutBinding.data();
//	vk::UniqueDescriptorSetLayout descSetLayout = device->createDescriptorSetLayoutUnique(descSetLayoutInfo);
//
//	// デスクリプタセットの作成.
//	vk::DescriptorSetAllocateInfo descSetInfo = {};
//	descSetInfo.descriptorPool = descPool.get();
//	descSetInfo.descriptorSetCount = 1;
//	descSetInfo.pSetLayouts = &descSetLayout.get();
//
//	vk::UniqueDescriptorSet descSet = std::move(device->allocateDescriptorSetsUnique(descSetInfo)[0]);
//
//	// コンピュートパイプラインの作成.
//	vk::PushConstantRange pushConstantRange = {};
//	pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
//	pushConstantRange.offset = 0;
//	pushConstantRange.size = sizeof(int) * 4;
//
//	vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
//	pipelineLayoutInfo.setLayoutCount = 1;
//	pipelineLayoutInfo.pSetLayouts = &descSetLayout.get();
//	pipelineLayoutInfo.pushConstantRangeCount = 1;
//	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
//
//	vk::UniquePipelineLayout pipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutInfo);
//
//	vk::ComputePipelineCreateInfo pipelineInfo = {};
//	pipelineInfo.stage.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
//	pipelineInfo.stage.stage = vk::ShaderStageFlagBits::eCompute;
//	pipelineInfo.stage.module = computeShaderModule.get();
//	pipelineInfo.stage.pName = "main";
//	pipelineInfo.layout = pipelineLayout.get();
//
//	vk::UniquePipelineCache pipelineCache = device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo());
//
//	auto wrapped = device->createComputePipelinesUnique(pipelineCache.get(), pipelineInfo);
//	vk::UniquePipeline computePipeline = std::move(wrapped.value[0]);
//
//	// デスクリプタにbind.
//	vk::DescriptorBufferInfo bufferInfoA = {};
//	bufferInfoA.buffer = bufferA.get();
//	bufferInfoA.offset = 0;
//	bufferInfoA.range = bufferSizeInA;
//
//	vk::DescriptorBufferInfo bufferInfoB = {};
//	bufferInfoB.buffer = bufferB.get();
//	bufferInfoB.offset = 0;
//	bufferInfoB.range = bufferSizeInB;
//
//	vk::DescriptorBufferInfo bufferInfoC = {};
//	bufferInfoC.buffer = bufferC.get();
//	bufferInfoC.offset = 0;
//	bufferInfoC.range = bufferSizeOut;
//
//	std::vector<vk::WriteDescriptorSet> descWrites(3);
//	descWrites[0].dstSet = descSet.get();
//	descWrites[0].dstBinding = 0;
//	descWrites[0].dstArrayElement = 0;
//	descWrites[0].descriptorCount = 1;
//	descWrites[0].descriptorType = vk::DescriptorType::eStorageBuffer;
//	descWrites[0].pBufferInfo = &bufferInfoA;
//
//
//	descWrites[1].dstSet = descSet.get();
//	descWrites[1].dstBinding = 1;
//	descWrites[1].dstArrayElement = 0;
//	descWrites[1].descriptorCount = 1;
//	descWrites[1].descriptorType = vk::DescriptorType::eStorageBuffer;
//	descWrites[1].pBufferInfo = &bufferInfoB;
//
//	descWrites[2].dstSet = descSet.get();
//	descWrites[2].dstBinding = 2;
//	descWrites[2].dstArrayElement = 0;
//	descWrites[2].descriptorCount = 1;	
//	descWrites[2].descriptorType = vk::DescriptorType::eStorageBuffer;
//	descWrites[2].pBufferInfo = &bufferInfoC;
//
//	device->updateDescriptorSets(descWrites, nullptr);
//
//	vk::CommandBufferAllocateInfo cmdAllocateInfo = {};
//	cmdAllocateInfo.level = vk::CommandBufferLevel::ePrimary;
//	cmdAllocateInfo.commandPool = commandPool.get();
//	cmdAllocateInfo.commandBufferCount = 1;
//
//	vk::UniqueCommandBuffer calcCommandBuffer = std::move(device->allocateCommandBuffersUnique(cmdAllocateInfo)[0]);
//
//	vk::CommandBufferBeginInfo beginInfo = {};
//	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
//	calcCommandBuffer->begin(beginInfo);
//	calcCommandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline.get());
//	calcCommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descSet.get(), {});
//
//	// プッシュコンスタントの設定.
//	struct pushdata {
//		int m_WidthA = WIDTHA;
//		int m_HeightA = HEIGHTA;
//		int m_WidthB = WIDTHB;
//		int m_HeightB = HEIGHTB;
//	} constantValue;
//
//	calcCommandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0, 
//		sizeof(constantValue), &constantValue);
//
//	int maxWidth = std::max(WIDTHA, WIDTHB);
//	int maxHeight = std::max(WIDTHA, WIDTHB);
//	calcCommandBuffer->dispatch((maxWidth + 15) / 16, (maxHeight + 15) / 16, 1);
//	calcCommandBuffer->end();
//
//	const vk::PipelineStageFlags stageCalc = vk::PipelineStageFlagBits::eComputeShader;
//
//	submitInfo.commandBufferCount = 1;
//	submitInfo.pCommandBuffers = &calcCommandBuffer.get();
//	submitInfo.waitSemaphoreCount = 1;
//	submitInfo.setPWaitSemaphores(&semaphore.get());
//	submitInfo.pWaitDstStageMask = &stageCalc;
//
//	queue.submit(submitInfo);
//
//	data = device->mapMemory(stagingBufferMemoryC.get(), 0, bufferSizeOut);
//	vk::UniqueCommandBuffer getCommandBuffer = makeCopyCommandBuffer(device.get(), commandPool.get(),
//		bufferC.get(), stagingBufferC.get(), bufferSizeOut, vk::CommandBufferLevel::ePrimary);
//
//	const vk::PipelineStageFlags stageTrans = vk::PipelineStageFlagBits::eTransfer;
//
//	submitInfo.commandBufferCount = 1;
//	submitInfo.pCommandBuffers = &getCommandBuffer.get();
//	submitInfo.waitSemaphoreCount = 1;
//	submitInfo.setPWaitSemaphores(&semaphore.get());
//	submitInfo.pWaitDstStageMask = &stageTrans;
//
//	queue.submit(submitInfo, fence.get());
//
//	const std::vector< vk::Fence > fences{ fence.get() };
//	if (device->waitForFences(fences, true, 10000000000u) != vk::Result::eSuccess) {
//		abort();
//	}
//
//	memcpy(dataC.data(), data, bufferSizeOut);
//	device->unmapMemory(stagingBufferMemoryC.get());
//
//
//	// 結果を出力.
//	for (int i = 0; i < 100; ++i) {
//		for (int j = 0; j < 100; ++j) {
//			matrixC[i][j] = dataC[i * WIDTHB + j];
//			std::cout << "row: " << i << " col: " << j << " value: " << matrixC[i][j] << std::endl;
//		}
//	}
//
//	return 0;
//}