#include "matCalculation.h"

Matrix::Matrix() {
	initVulkan();
};

Matrix::~Matrix() {};

uint32_t Matrix::findMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags property) {
	vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if (memoryProperties.memoryTypes[i].propertyFlags & property) {
			if (typeFilter & (1 << i)) {
				return i;
			}
		}
	}

	throw std::runtime_error("failed to findMemoryType.");
}

vk::UniqueBuffer Matrix::createBuffer(const vk::Device device, const vk::PhysicalDevice physicalDevice, const vk::DeviceSize bufferSize,
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

vk::UniqueCommandBuffer Matrix::makeCopyCommandBuffer(const vk::Device device, vk::CommandPool commandPool, vk::Buffer& srcBuffer, vk::Buffer& dstBuffer,
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

std::vector<char> Matrix::readFile(const std::string& filename) {
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

vk::UniqueShaderModule Matrix::createShaderModule(const vk::Device device, const std::vector<char>& code) {
	vk::ShaderModuleCreateInfo shaderInfo = {};
	shaderInfo.codeSize = code.size();
	shaderInfo.pCode = reinterpret_cast<const uint32_t*> (code.data());
	return std::move(device.createShaderModuleUnique(shaderInfo));
}

void Matrix::initVulkan() {
	vk::ApplicationInfo appInfo = {};
	appInfo.pApplicationName = "MatrixCalculation";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "No Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	vk::InstanceCreateInfo instanceInfo = {};

	// バリデーションレイヤーの使用.
	const std::vector<const char*> layers = {
		"VK_LAYER_KHRONOS_validation"
	};
	instanceInfo.pApplicationInfo = &appInfo;
	instanceInfo.ppEnabledLayerNames = layers.data();
	instanceInfo.enabledLayerCount = static_cast<uint32_t> (layers.size());

	m_instance = vk::createInstanceUnique(instanceInfo);

	m_physicalDevice = m_instance->enumeratePhysicalDevices()[0];

	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = m_physicalDevice.getQueueFamilyProperties();
	uint32_t queueFamilyIndex = -1;
	for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
		if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) {
			queueFamilyIndex = i;
			break;
		}
	}

	if (queueFamilyIndex == -1) {
		throw std::runtime_error("failed to find queueFamilyIndex.");
	}

	// 論理デバイスとキューの作成.
	float queuePriority = 0.0f;
	vk::DeviceQueueCreateInfo queueCreateInfo = {};
	queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	std::vector<vk::DeviceQueueCreateInfo> queues{ queueCreateInfo };

	vk::DeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t> (queues.size());
	deviceCreateInfo.pQueueCreateInfos = queues.data();
	deviceCreateInfo.enabledLayerCount = static_cast<uint32_t> (layers.size());
	deviceCreateInfo.ppEnabledLayerNames = layers.data();

	m_device = m_physicalDevice.createDeviceUnique(deviceCreateInfo);
	m_queue = m_device->getQueue(queueFamilyIndex, 0);

	// セマフォとフェンスの作成.
	m_semaphore = m_device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
	m_fence = m_device->createFenceUnique({});

	// コマンドプールの作成.
	vk::CommandPoolCreateInfo poolCreateInfo = {};
	poolCreateInfo.queueFamilyIndex = queueFamilyIndex;
	m_commandPool = m_device->createCommandPoolUnique(poolCreateInfo);

	// スレッドのサイズの表示.
	vk::PhysicalDeviceProperties physicalProp = m_physicalDevice.getProperties();

	std::cout << "max compute work group num: "
		<< physicalProp.limits.maxComputeWorkGroupCount[0] << ", "
		<< physicalProp.limits.maxComputeWorkGroupCount[1] << ", "
		<< physicalProp.limits.maxComputeWorkGroupCount[2] << std::endl;

	std::cout << "max compute work group size: "
		<< physicalProp.limits.maxComputeWorkGroupSize[0] << ", "
		<< physicalProp.limits.maxComputeWorkGroupSize[1] << ", "
		<< physicalProp.limits.maxComputeWorkGroupSize[2] << std::endl;

	std::cout << "max thread size: "
		<< physicalProp.limits.maxComputeWorkGroupInvocations << std::endl;
}



std::vector<std::vector<float>> Matrix::multi(const std::vector<std::vector<float>> &matA, const std::vector<std::vector<float>> &matB) {
	if (matA.empty() || matB.empty()) {
		throw std::runtime_error("matrix is empty.");
		abort();
	}

	int heightA = (int)(matA.size());
	int widthA = (int)(matA[0].size());
	int heightB = (int)(matB.size());
	int widthB = (int)(matB[0].size());

	if (widthA != heightB) {
		throw std::runtime_error("don't match matrix size for calculation.");
		abort();
	}

	// バッファの作成.
	std::vector<std::vector<float>> ans(heightA, std::vector<float>(widthB));

	std::vector<float> dataA(widthA * heightA);
	std::vector<float> dataB(widthB* heightB);
	std::vector<float> dataC(widthB* heightA);

	for (int i = 0; i < heightA; ++i) {
		std::copy(matA[i].begin(), matA[i].end(), dataA.begin() + i * widthA);
	}

	for (int i = 0; i < heightB; ++i) {
		std::copy(matB[i].begin(), matB[i].end(), dataB.begin() + i * widthB);
	}

	vk::DeviceSize bufferSizeInA = sizeof(float) * widthA * heightA;
	vk::DeviceSize bufferSizeInB = sizeof(float) * widthB * heightB;
	vk::DeviceSize bufferSizeOut = sizeof(float) * heightA * widthB;

	vk::UniqueDeviceMemory stagingBufferMemoryA, stagingBufferMemoryB, stagingBufferMemoryC, bufferAMemory, bufferBMemory, bufferCMemory;

	vk::UniqueBuffer stagingBufferA = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInA,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryA);

	vk::UniqueBuffer stagingBufferB = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInB,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryB);

	vk::UniqueBuffer stagingBufferC = createBuffer(m_device.get(), m_physicalDevice, bufferSizeOut,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryC);

	vk::UniqueBuffer bufferA = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInA,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferAMemory);

	vk::UniqueBuffer bufferB = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInB,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferBMemory);

	vk::UniqueBuffer bufferC = createBuffer(m_device.get(), m_physicalDevice, bufferSizeOut,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferCMemory);

	// コマンドバッファの作成.
	void* data;
	data = m_device->mapMemory(stagingBufferMemoryA.get(), 0, bufferSizeInA);
	std::memcpy(data, dataA.data(), bufferSizeInA);
	vk::UniqueCommandBuffer commandBufferA = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		stagingBufferA.get(), bufferA.get(), bufferSizeInA, vk::CommandBufferLevel::ePrimary);
	m_device->unmapMemory(stagingBufferMemoryA.get());

	data = m_device->mapMemory(stagingBufferMemoryB.get(), 0, bufferSizeInB);
	std::memcpy(data, dataB.data(), bufferSizeInB);
	vk::UniqueCommandBuffer commandBufferB = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		stagingBufferB.get(), bufferB.get(), bufferSizeInB, vk::CommandBufferLevel::ePrimary);
	m_device->unmapMemory(stagingBufferMemoryB.get());

	std::vector<vk::CommandBuffer> commandBuffers = { commandBufferA.get(), commandBufferB.get() };

	// キューに流して実行.
	vk::SubmitInfo submitInfo = {};
	submitInfo.commandBufferCount = static_cast<uint32_t> (commandBuffers.size());
	submitInfo.pCommandBuffers = commandBuffers.data();
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.setPSignalSemaphores(&m_semaphore.get());
	m_queue.submit(submitInfo);

	// シェーダモジュールの設定. 修正必要.
	std::vector<char> code = readFile("src/matMultiple.spv");
	vk::UniqueShaderModule computeShaderModule = createShaderModule(m_device.get(), code);

	// デスクリプタプールの作成.
	vk::DescriptorPoolSize descPoolSize = {};
	descPoolSize.type = vk::DescriptorType::eStorageBuffer;
	descPoolSize.descriptorCount = 3; // デスクリプタに結び付けるバッファの数.

	vk::DescriptorPoolCreateInfo descPoolInfo = {};
	descPoolInfo.poolSizeCount = 1;
	descPoolInfo.pPoolSizes = &descPoolSize;
	descPoolInfo.maxSets = 1;
	descPoolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	vk::UniqueDescriptorPool descPool = m_device->createDescriptorPoolUnique(descPoolInfo);

	// デスクリプタセットレイアウトの作成.
	std::vector<vk::DescriptorSetLayoutBinding> descSetLayoutBinding(3);
	descSetLayoutBinding[0].binding = 0;
	descSetLayoutBinding[0].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[0].descriptorCount = 1;
	descSetLayoutBinding[0].stageFlags = vk::ShaderStageFlagBits::eCompute;

	descSetLayoutBinding[1].binding = 1;
	descSetLayoutBinding[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[1].descriptorCount = 1;
	descSetLayoutBinding[1].stageFlags = vk::ShaderStageFlagBits::eCompute;

	descSetLayoutBinding[2].binding = 2;
	descSetLayoutBinding[2].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[2].descriptorCount = 1;
	descSetLayoutBinding[2].stageFlags = vk::ShaderStageFlagBits::eCompute;

	vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo = {};
	descSetLayoutInfo.bindingCount = static_cast<uint32_t> (descSetLayoutBinding.size());
	descSetLayoutInfo.pBindings = descSetLayoutBinding.data();
	vk::UniqueDescriptorSetLayout descSetLayout = m_device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

	// デスクリプタセットの作成.
	vk::DescriptorSetAllocateInfo descSetInfo = {};
	descSetInfo.descriptorPool = descPool.get();
	descSetInfo.descriptorSetCount = 1;
	descSetInfo.pSetLayouts = &descSetLayout.get();

	vk::UniqueDescriptorSet descSet = std::move(m_device->allocateDescriptorSetsUnique(descSetInfo)[0]);

	// コンピュートパイプラインの作成.
	vk::PushConstantRange pushConstantRange = {};
	pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(int) * 4;

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descSetLayout.get();
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

	vk::UniquePipelineLayout pipelineLayout = m_device->createPipelineLayoutUnique(pipelineLayoutInfo);

	vk::ComputePipelineCreateInfo pipelineInfo = {};
	pipelineInfo.stage.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
	pipelineInfo.stage.stage = vk::ShaderStageFlagBits::eCompute;
	pipelineInfo.stage.module = computeShaderModule.get();
	pipelineInfo.stage.pName = "main";
	pipelineInfo.layout = pipelineLayout.get();

	vk::UniquePipelineCache pipelineCache = m_device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo());

	auto wrapped = m_device->createComputePipelinesUnique(pipelineCache.get(), pipelineInfo);
	vk::UniquePipeline computePipeline = std::move(wrapped.value[0]);

	// デスクリプタにbind.
	vk::DescriptorBufferInfo bufferInfoA = {};
	bufferInfoA.buffer = bufferA.get();
	bufferInfoA.offset = 0;
	bufferInfoA.range = bufferSizeInA;

	vk::DescriptorBufferInfo bufferInfoB = {};
	bufferInfoB.buffer = bufferB.get();
	bufferInfoB.offset = 0;
	bufferInfoB.range = bufferSizeInB;

	vk::DescriptorBufferInfo bufferInfoC = {};
	bufferInfoC.buffer = bufferC.get();
	bufferInfoC.offset = 0;
	bufferInfoC.range = bufferSizeOut;

	std::vector<vk::WriteDescriptorSet> descWrites(3);
	descWrites[0].dstSet = descSet.get();
	descWrites[0].dstBinding = 0;
	descWrites[0].dstArrayElement = 0;
	descWrites[0].descriptorCount = 1;
	descWrites[0].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[0].pBufferInfo = &bufferInfoA;


	descWrites[1].dstSet = descSet.get();
	descWrites[1].dstBinding = 1;
	descWrites[1].dstArrayElement = 0;
	descWrites[1].descriptorCount = 1;
	descWrites[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[1].pBufferInfo = &bufferInfoB;

	descWrites[2].dstSet = descSet.get();
	descWrites[2].dstBinding = 2;
	descWrites[2].dstArrayElement = 0;
	descWrites[2].descriptorCount = 1;
	descWrites[2].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[2].pBufferInfo = &bufferInfoC;

	m_device->updateDescriptorSets(descWrites, nullptr);

	vk::CommandBufferAllocateInfo cmdAllocateInfo = {};
	cmdAllocateInfo.level = vk::CommandBufferLevel::ePrimary;
	cmdAllocateInfo.commandPool = m_commandPool.get();
	cmdAllocateInfo.commandBufferCount = 1;

	vk::UniqueCommandBuffer calcCommandBuffer = std::move(m_device->allocateCommandBuffersUnique(cmdAllocateInfo)[0]);

	vk::CommandBufferBeginInfo beginInfo = {};
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	calcCommandBuffer->begin(beginInfo);
	calcCommandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline.get());
	calcCommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descSet.get(), {});

	// プッシュコンスタントの設定.
	struct pushdata {
		int m_WidthA;
		int m_HeightA;
		int m_WidthB;
		int m_HeightB;
	} constantValue;

	constantValue.m_WidthA = widthA;
	constantValue.m_HeightA = heightA;
	constantValue.m_WidthB = widthB;
	constantValue.m_HeightB = heightB;


	calcCommandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0,
		sizeof(constantValue), &constantValue);

	int maxWidth = std::max(widthA, widthB);
	int maxHeight = std::max(heightA, heightB);
	calcCommandBuffer->dispatch((maxWidth + 15) / 16, (maxHeight + 15) / 16, 1);
	calcCommandBuffer->end();

	const vk::PipelineStageFlags stageCalc = vk::PipelineStageFlagBits::eComputeShader;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &calcCommandBuffer.get();
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.setPWaitSemaphores(&m_semaphore.get());
	submitInfo.pWaitDstStageMask = &stageCalc;

	m_queue.submit(submitInfo);

	data = m_device->mapMemory(stagingBufferMemoryC.get(), 0, bufferSizeOut);
	vk::UniqueCommandBuffer getCommandBuffer = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		bufferC.get(), stagingBufferC.get(), bufferSizeOut, vk::CommandBufferLevel::ePrimary);

	const vk::PipelineStageFlags stageTrans = vk::PipelineStageFlagBits::eTransfer;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &getCommandBuffer.get();
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.setPWaitSemaphores(&m_semaphore.get());
	submitInfo.pWaitDstStageMask = &stageTrans;

	m_queue.submit(submitInfo, m_fence.get());

	const std::vector< vk::Fence > fences{ m_fence.get() };
	if (m_device->waitForFences(fences, true, 10000000000u) != vk::Result::eSuccess) {
		abort();
	}

	memcpy(dataC.data(), data, bufferSizeOut);
	m_device->unmapMemory(stagingBufferMemoryC.get());


	for (int i = 0; i < heightA; ++i) {
		std::copy(dataC.begin() + i * widthB, dataC.begin() + (i + 1) * widthB, ans[i].begin());
	}

	return ans;
}

std::vector<std::vector<float>> Matrix::multiEach(const std::vector<std::vector<float>>& matA, const std::vector<std::vector<float>>& matB) {
	if (matA.empty() || matB.empty()) {
		throw std::runtime_error("matrix is empty.");
		abort();
	}

	int heightA = (int)(matA.size());
	int widthA = (int)(matA[0].size());
	int heightB = (int)(matB.size());
	int widthB = (int)(matB[0].size());

	if ((widthA != widthB) || (heightA != heightB)) {
		throw std::runtime_error("don't match matrix size for calculation.");
		abort();
	}

	int width = widthA;
	int height = heightA;

	// バッファの作成.
	std::vector<std::vector<float>> ans(height, std::vector<float>(width));

	std::vector<float> dataA(width * height);
	std::vector<float> dataB(width * height);
	std::vector<float> dataC(width * height);

	for (int i = 0; i < height; ++i) {
		std::copy(matA[i].begin(), matA[i].end(), dataA.begin() + i * width);
	}

	for (int i = 0; i < height; ++i) {
		std::copy(matB[i].begin(), matB[i].end(), dataB.begin() + i * width);
	}

	vk::DeviceSize bufferSizeInA = sizeof(float) * width * height;
	vk::DeviceSize bufferSizeInB = sizeof(float) * width * height;
	vk::DeviceSize bufferSizeOut = sizeof(float) * height * width;

	vk::UniqueDeviceMemory stagingBufferMemoryA, stagingBufferMemoryB, stagingBufferMemoryC, bufferAMemory, bufferBMemory, bufferCMemory;

	vk::UniqueBuffer stagingBufferA = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInA,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryA);

	vk::UniqueBuffer stagingBufferB = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInB,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryB);

	vk::UniqueBuffer stagingBufferC = createBuffer(m_device.get(), m_physicalDevice, bufferSizeOut,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryC);

	vk::UniqueBuffer bufferA = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInA,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferAMemory);

	vk::UniqueBuffer bufferB = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInB,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferBMemory);

	vk::UniqueBuffer bufferC = createBuffer(m_device.get(), m_physicalDevice, bufferSizeOut,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferCMemory);

	// コマンドバッファの作成.
	void* data;
	data = m_device->mapMemory(stagingBufferMemoryA.get(), 0, bufferSizeInA);
	std::memcpy(data, dataA.data(), bufferSizeInA);
	vk::UniqueCommandBuffer commandBufferA = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		stagingBufferA.get(), bufferA.get(), bufferSizeInA, vk::CommandBufferLevel::ePrimary);
	m_device->unmapMemory(stagingBufferMemoryA.get());

	data = m_device->mapMemory(stagingBufferMemoryB.get(), 0, bufferSizeInB);
	std::memcpy(data, dataB.data(), bufferSizeInB);
	vk::UniqueCommandBuffer commandBufferB = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		stagingBufferB.get(), bufferB.get(), bufferSizeInB, vk::CommandBufferLevel::ePrimary);
	m_device->unmapMemory(stagingBufferMemoryB.get());

	std::vector<vk::CommandBuffer> commandBuffers = { commandBufferA.get(), commandBufferB.get() };

	// キューに流して実行.
	vk::SubmitInfo submitInfo = {};
	submitInfo.commandBufferCount = static_cast<uint32_t> (commandBuffers.size());
	submitInfo.pCommandBuffers = commandBuffers.data();
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.setPSignalSemaphores(&m_semaphore.get());
	m_queue.submit(submitInfo);

	// シェーダモジュールの設定. 修正必要.
	std::vector<char> code = readFile("src/matMutipleEach.spv");
	vk::UniqueShaderModule computeShaderModule = createShaderModule(m_device.get(), code);

	// デスクリプタプールの作成.
	vk::DescriptorPoolSize descPoolSize = {};
	descPoolSize.type = vk::DescriptorType::eStorageBuffer;
	descPoolSize.descriptorCount = 3; // デスクリプタに結び付けるバッファの数.

	vk::DescriptorPoolCreateInfo descPoolInfo = {};
	descPoolInfo.poolSizeCount = 1;
	descPoolInfo.pPoolSizes = &descPoolSize;
	descPoolInfo.maxSets = 1;
	descPoolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	vk::UniqueDescriptorPool descPool = m_device->createDescriptorPoolUnique(descPoolInfo);

	// デスクリプタセットレイアウトの作成.
	std::vector<vk::DescriptorSetLayoutBinding> descSetLayoutBinding(3);
	descSetLayoutBinding[0].binding = 0;
	descSetLayoutBinding[0].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[0].descriptorCount = 1;
	descSetLayoutBinding[0].stageFlags = vk::ShaderStageFlagBits::eCompute;

	descSetLayoutBinding[1].binding = 1;
	descSetLayoutBinding[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[1].descriptorCount = 1;
	descSetLayoutBinding[1].stageFlags = vk::ShaderStageFlagBits::eCompute;

	descSetLayoutBinding[2].binding = 2;
	descSetLayoutBinding[2].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[2].descriptorCount = 1;
	descSetLayoutBinding[2].stageFlags = vk::ShaderStageFlagBits::eCompute;

	vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo = {};
	descSetLayoutInfo.bindingCount = static_cast<uint32_t> (descSetLayoutBinding.size());
	descSetLayoutInfo.pBindings = descSetLayoutBinding.data();
	vk::UniqueDescriptorSetLayout descSetLayout = m_device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

	// デスクリプタセットの作成.
	vk::DescriptorSetAllocateInfo descSetInfo = {};
	descSetInfo.descriptorPool = descPool.get();
	descSetInfo.descriptorSetCount = 1;
	descSetInfo.pSetLayouts = &descSetLayout.get();

	vk::UniqueDescriptorSet descSet = std::move(m_device->allocateDescriptorSetsUnique(descSetInfo)[0]);

	// コンピュートパイプラインの作成.
	vk::PushConstantRange pushConstantRange = {};
	pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(int) * 2;

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descSetLayout.get();
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

	vk::UniquePipelineLayout pipelineLayout = m_device->createPipelineLayoutUnique(pipelineLayoutInfo);

	vk::ComputePipelineCreateInfo pipelineInfo = {};
	pipelineInfo.stage.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
	pipelineInfo.stage.stage = vk::ShaderStageFlagBits::eCompute;
	pipelineInfo.stage.module = computeShaderModule.get();
	pipelineInfo.stage.pName = "main";
	pipelineInfo.layout = pipelineLayout.get();

	vk::UniquePipelineCache pipelineCache = m_device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo());

	auto wrapped = m_device->createComputePipelinesUnique(pipelineCache.get(), pipelineInfo);
	vk::UniquePipeline computePipeline = std::move(wrapped.value[0]);

	// デスクリプタにbind.
	vk::DescriptorBufferInfo bufferInfoA = {};
	bufferInfoA.buffer = bufferA.get();
	bufferInfoA.offset = 0;
	bufferInfoA.range = bufferSizeInA;

	vk::DescriptorBufferInfo bufferInfoB = {};
	bufferInfoB.buffer = bufferB.get();
	bufferInfoB.offset = 0;
	bufferInfoB.range = bufferSizeInB;

	vk::DescriptorBufferInfo bufferInfoC = {};
	bufferInfoC.buffer = bufferC.get();
	bufferInfoC.offset = 0;
	bufferInfoC.range = bufferSizeOut;

	std::vector<vk::WriteDescriptorSet> descWrites(3);
	descWrites[0].dstSet = descSet.get();
	descWrites[0].dstBinding = 0;
	descWrites[0].dstArrayElement = 0;
	descWrites[0].descriptorCount = 1;
	descWrites[0].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[0].pBufferInfo = &bufferInfoA;


	descWrites[1].dstSet = descSet.get();
	descWrites[1].dstBinding = 1;
	descWrites[1].dstArrayElement = 0;
	descWrites[1].descriptorCount = 1;
	descWrites[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[1].pBufferInfo = &bufferInfoB;

	descWrites[2].dstSet = descSet.get();
	descWrites[2].dstBinding = 2;
	descWrites[2].dstArrayElement = 0;
	descWrites[2].descriptorCount = 1;
	descWrites[2].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[2].pBufferInfo = &bufferInfoC;

	m_device->updateDescriptorSets(descWrites, nullptr);

	vk::CommandBufferAllocateInfo cmdAllocateInfo = {};
	cmdAllocateInfo.level = vk::CommandBufferLevel::ePrimary;
	cmdAllocateInfo.commandPool = m_commandPool.get();
	cmdAllocateInfo.commandBufferCount = 1;

	vk::UniqueCommandBuffer calcCommandBuffer = std::move(m_device->allocateCommandBuffersUnique(cmdAllocateInfo)[0]);

	vk::CommandBufferBeginInfo beginInfo = {};
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	calcCommandBuffer->begin(beginInfo);
	calcCommandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline.get());
	calcCommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descSet.get(), {});

	// プッシュコンスタントの設定.
	struct pushdata {
		int width;
		int height;
	} constantValue;

	constantValue.width = width;
	constantValue.height = height;


	calcCommandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0,
		sizeof(constantValue), &constantValue);

	calcCommandBuffer->dispatch((width + 15) / 16, (height + 15) / 16, 1);
	calcCommandBuffer->end();

	const vk::PipelineStageFlags stageCalc = vk::PipelineStageFlagBits::eComputeShader;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &calcCommandBuffer.get();
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.setPWaitSemaphores(&m_semaphore.get());
	submitInfo.pWaitDstStageMask = &stageCalc;

	m_queue.submit(submitInfo);

	data = m_device->mapMemory(stagingBufferMemoryC.get(), 0, bufferSizeOut);
	vk::UniqueCommandBuffer getCommandBuffer = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		bufferC.get(), stagingBufferC.get(), bufferSizeOut, vk::CommandBufferLevel::ePrimary);

	const vk::PipelineStageFlags stageTrans = vk::PipelineStageFlagBits::eTransfer;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &getCommandBuffer.get();
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.setPWaitSemaphores(&m_semaphore.get());
	submitInfo.pWaitDstStageMask = &stageTrans;

	m_queue.submit(submitInfo, m_fence.get());

	const std::vector< vk::Fence > fences{ m_fence.get() };
	if (m_device->waitForFences(fences, true, 10000000000u) != vk::Result::eSuccess) {
		abort();
	}

	memcpy(dataC.data(), data, bufferSizeOut);
	m_device->unmapMemory(stagingBufferMemoryC.get());


	for (int i = 0; i < height; ++i) {
		std::copy(dataC.begin() + i * width, dataC.begin() + (i + 1) * width, ans[i].begin());
	}

	return ans;
}

std::vector<std::vector<float>> Matrix::sum(const std::vector<std::vector<float>>& matA, const std::vector<std::vector<float>>& matB) {
	if (matA.empty() || matB.empty()) {
		throw std::runtime_error("matrix is empty.");
		abort();
	}

	int heightA = (int)(matA.size());
	int widthA = (int)(matA[0].size());
	int heightB = (int)(matB.size());
	int widthB = (int)(matB[0].size());

	if ((widthA != widthB) || (heightA != heightB)) {
		throw std::runtime_error("don't match matrix size for calculation.");
		abort();
	}

	int width = widthA;
	int height = heightA;

	// バッファの作成.
	std::vector<std::vector<float>> ans(height, std::vector<float>(width));

	std::vector<float> dataA(width * height);
	std::vector<float> dataB(width * height);
	std::vector<float> dataC(width * height);

	for (int i = 0; i < height; ++i) {
		std::copy(matA[i].begin(), matA[i].end(), dataA.begin() + i * width);
	}

	for (int i = 0; i < height; ++i) {
		std::copy(matB[i].begin(), matB[i].end(), dataB.begin() + i * width);
	}

	vk::DeviceSize bufferSizeInA = sizeof(float) * width * height;
	vk::DeviceSize bufferSizeInB = sizeof(float) * width * height;
	vk::DeviceSize bufferSizeOut = sizeof(float) * height * width;

	vk::UniqueDeviceMemory stagingBufferMemoryA, stagingBufferMemoryB, stagingBufferMemoryC, bufferAMemory, bufferBMemory, bufferCMemory;

	vk::UniqueBuffer stagingBufferA = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInA,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryA);

	vk::UniqueBuffer stagingBufferB = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInB,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryB);

	vk::UniqueBuffer stagingBufferC = createBuffer(m_device.get(), m_physicalDevice, bufferSizeOut,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryC);

	vk::UniqueBuffer bufferA = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInA,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferAMemory);

	vk::UniqueBuffer bufferB = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInB,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferBMemory);

	vk::UniqueBuffer bufferC = createBuffer(m_device.get(), m_physicalDevice, bufferSizeOut,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferCMemory);

	// コマンドバッファの作成.
	void* data;
	data = m_device->mapMemory(stagingBufferMemoryA.get(), 0, bufferSizeInA);
	std::memcpy(data, dataA.data(), bufferSizeInA);
	vk::UniqueCommandBuffer commandBufferA = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		stagingBufferA.get(), bufferA.get(), bufferSizeInA, vk::CommandBufferLevel::ePrimary);
	m_device->unmapMemory(stagingBufferMemoryA.get());

	data = m_device->mapMemory(stagingBufferMemoryB.get(), 0, bufferSizeInB);
	std::memcpy(data, dataB.data(), bufferSizeInB);
	vk::UniqueCommandBuffer commandBufferB = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		stagingBufferB.get(), bufferB.get(), bufferSizeInB, vk::CommandBufferLevel::ePrimary);
	m_device->unmapMemory(stagingBufferMemoryB.get());

	std::vector<vk::CommandBuffer> commandBuffers = { commandBufferA.get(), commandBufferB.get() };

	// キューに流して実行.
	vk::SubmitInfo submitInfo = {};
	submitInfo.commandBufferCount = static_cast<uint32_t> (commandBuffers.size());
	submitInfo.pCommandBuffers = commandBuffers.data();
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.setPSignalSemaphores(&m_semaphore.get());
	m_queue.submit(submitInfo);

	// シェーダモジュールの設定. 修正必要.
	std::vector<char> code = readFile("src/matSum.spv");
	vk::UniqueShaderModule computeShaderModule = createShaderModule(m_device.get(), code);

	// デスクリプタプールの作成.
	vk::DescriptorPoolSize descPoolSize = {};
	descPoolSize.type = vk::DescriptorType::eStorageBuffer;
	descPoolSize.descriptorCount = 3; // デスクリプタに結び付けるバッファの数.

	vk::DescriptorPoolCreateInfo descPoolInfo = {};
	descPoolInfo.poolSizeCount = 1;
	descPoolInfo.pPoolSizes = &descPoolSize;
	descPoolInfo.maxSets = 1;
	descPoolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	vk::UniqueDescriptorPool descPool = m_device->createDescriptorPoolUnique(descPoolInfo);

	// デスクリプタセットレイアウトの作成.
	std::vector<vk::DescriptorSetLayoutBinding> descSetLayoutBinding(3);
	descSetLayoutBinding[0].binding = 0;
	descSetLayoutBinding[0].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[0].descriptorCount = 1;
	descSetLayoutBinding[0].stageFlags = vk::ShaderStageFlagBits::eCompute;

	descSetLayoutBinding[1].binding = 1;
	descSetLayoutBinding[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[1].descriptorCount = 1;
	descSetLayoutBinding[1].stageFlags = vk::ShaderStageFlagBits::eCompute;

	descSetLayoutBinding[2].binding = 2;
	descSetLayoutBinding[2].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[2].descriptorCount = 1;
	descSetLayoutBinding[2].stageFlags = vk::ShaderStageFlagBits::eCompute;

	vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo = {};
	descSetLayoutInfo.bindingCount = static_cast<uint32_t> (descSetLayoutBinding.size());
	descSetLayoutInfo.pBindings = descSetLayoutBinding.data();
	vk::UniqueDescriptorSetLayout descSetLayout = m_device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

	// デスクリプタセットの作成.
	vk::DescriptorSetAllocateInfo descSetInfo = {};
	descSetInfo.descriptorPool = descPool.get();
	descSetInfo.descriptorSetCount = 1;
	descSetInfo.pSetLayouts = &descSetLayout.get();

	vk::UniqueDescriptorSet descSet = std::move(m_device->allocateDescriptorSetsUnique(descSetInfo)[0]);

	// コンピュートパイプラインの作成.
	vk::PushConstantRange pushConstantRange = {};
	pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(int) * 2;

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descSetLayout.get();
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

	vk::UniquePipelineLayout pipelineLayout = m_device->createPipelineLayoutUnique(pipelineLayoutInfo);

	vk::ComputePipelineCreateInfo pipelineInfo = {};
	pipelineInfo.stage.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
	pipelineInfo.stage.stage = vk::ShaderStageFlagBits::eCompute;
	pipelineInfo.stage.module = computeShaderModule.get();
	pipelineInfo.stage.pName = "main";
	pipelineInfo.layout = pipelineLayout.get();

	vk::UniquePipelineCache pipelineCache = m_device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo());

	auto wrapped = m_device->createComputePipelinesUnique(pipelineCache.get(), pipelineInfo);
	vk::UniquePipeline computePipeline = std::move(wrapped.value[0]);

	// デスクリプタにbind.
	vk::DescriptorBufferInfo bufferInfoA = {};
	bufferInfoA.buffer = bufferA.get();
	bufferInfoA.offset = 0;
	bufferInfoA.range = bufferSizeInA;

	vk::DescriptorBufferInfo bufferInfoB = {};
	bufferInfoB.buffer = bufferB.get();
	bufferInfoB.offset = 0;
	bufferInfoB.range = bufferSizeInB;

	vk::DescriptorBufferInfo bufferInfoC = {};
	bufferInfoC.buffer = bufferC.get();
	bufferInfoC.offset = 0;
	bufferInfoC.range = bufferSizeOut;

	std::vector<vk::WriteDescriptorSet> descWrites(3);
	descWrites[0].dstSet = descSet.get();
	descWrites[0].dstBinding = 0;
	descWrites[0].dstArrayElement = 0;
	descWrites[0].descriptorCount = 1;
	descWrites[0].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[0].pBufferInfo = &bufferInfoA;


	descWrites[1].dstSet = descSet.get();
	descWrites[1].dstBinding = 1;
	descWrites[1].dstArrayElement = 0;
	descWrites[1].descriptorCount = 1;
	descWrites[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[1].pBufferInfo = &bufferInfoB;

	descWrites[2].dstSet = descSet.get();
	descWrites[2].dstBinding = 2;
	descWrites[2].dstArrayElement = 0;
	descWrites[2].descriptorCount = 1;
	descWrites[2].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[2].pBufferInfo = &bufferInfoC;

	m_device->updateDescriptorSets(descWrites, nullptr);

	vk::CommandBufferAllocateInfo cmdAllocateInfo = {};
	cmdAllocateInfo.level = vk::CommandBufferLevel::ePrimary;
	cmdAllocateInfo.commandPool = m_commandPool.get();
	cmdAllocateInfo.commandBufferCount = 1;

	vk::UniqueCommandBuffer calcCommandBuffer = std::move(m_device->allocateCommandBuffersUnique(cmdAllocateInfo)[0]);

	vk::CommandBufferBeginInfo beginInfo = {};
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	calcCommandBuffer->begin(beginInfo);
	calcCommandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline.get());
	calcCommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descSet.get(), {});

	// プッシュコンスタントの設定.
	struct pushdata {
		int width;
		int height;
	} constantValue;

	constantValue.width = width;
	constantValue.height = height;


	calcCommandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0,
		sizeof(constantValue), &constantValue);

	calcCommandBuffer->dispatch((width + 15) / 16, (height + 15) / 16, 1);
	calcCommandBuffer->end();

	const vk::PipelineStageFlags stageCalc = vk::PipelineStageFlagBits::eComputeShader;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &calcCommandBuffer.get();
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.setPWaitSemaphores(&m_semaphore.get());
	submitInfo.pWaitDstStageMask = &stageCalc;

	m_queue.submit(submitInfo);

	data = m_device->mapMemory(stagingBufferMemoryC.get(), 0, bufferSizeOut);
	vk::UniqueCommandBuffer getCommandBuffer = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		bufferC.get(), stagingBufferC.get(), bufferSizeOut, vk::CommandBufferLevel::ePrimary);

	const vk::PipelineStageFlags stageTrans = vk::PipelineStageFlagBits::eTransfer;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &getCommandBuffer.get();
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.setPWaitSemaphores(&m_semaphore.get());
	submitInfo.pWaitDstStageMask = &stageTrans;

	m_queue.submit(submitInfo, m_fence.get());

	const std::vector< vk::Fence > fences{ m_fence.get() };
	if (m_device->waitForFences(fences, true, 10000000000u) != vk::Result::eSuccess) {
		abort();
	}

	memcpy(dataC.data(), data, bufferSizeOut);
	m_device->unmapMemory(stagingBufferMemoryC.get());


	for (int i = 0; i < height; ++i) {
		std::copy(dataC.begin() + i * width, dataC.begin() + (i + 1) * width, ans[i].begin());
	}

	return ans;
}

std::vector<std::vector<float>> Matrix::sum(const std::vector<std::vector<float>>& matA, const float value) {
	int height = (int)(matA.size());
	int width = (int)(matA[0].size());

	// バッファの作成.
	std::vector<std::vector<float>> ans(height, std::vector<float>(width));

	std::vector<float> dataA(width * height);
	std::vector<float> dataC(width * height);

	for (int i = 0; i < height; ++i) {
		std::copy(matA[i].begin(), matA[i].end(), dataA.begin() + i * width);
	}


	vk::DeviceSize bufferSizeInA = sizeof(float) * width * height;
	vk::DeviceSize bufferSizeOut = sizeof(float) * height * width;

	vk::UniqueDeviceMemory stagingBufferMemoryA, stagingBufferMemoryC, bufferAMemory, bufferCMemory;

	vk::UniqueBuffer stagingBufferA = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInA,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryA);

	vk::UniqueBuffer stagingBufferC = createBuffer(m_device.get(), m_physicalDevice, bufferSizeOut,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryC);

	vk::UniqueBuffer bufferA = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInA,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferAMemory);

	vk::UniqueBuffer bufferC = createBuffer(m_device.get(), m_physicalDevice, bufferSizeOut,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferCMemory);

	// コマンドバッファの作成.
	void* data;
	data = m_device->mapMemory(stagingBufferMemoryA.get(), 0, bufferSizeInA);
	std::memcpy(data, dataA.data(), bufferSizeInA);
	vk::UniqueCommandBuffer commandBufferA = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		stagingBufferA.get(), bufferA.get(), bufferSizeInA, vk::CommandBufferLevel::ePrimary);
	m_device->unmapMemory(stagingBufferMemoryA.get());

	std::vector<vk::CommandBuffer> commandBuffers = { commandBufferA.get() };

	// キューに流して実行.
	vk::SubmitInfo submitInfo = {};
	submitInfo.commandBufferCount = static_cast<uint32_t> (commandBuffers.size());
	submitInfo.pCommandBuffers = commandBuffers.data();
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.setPSignalSemaphores(&m_semaphore.get());
	m_queue.submit(submitInfo);

	// シェーダモジュールの設定. 修正必要.
	std::vector<char> code = readFile("src/matConstantSum.spv");
	vk::UniqueShaderModule computeShaderModule = createShaderModule(m_device.get(), code);

	// デスクリプタプールの作成.
	vk::DescriptorPoolSize descPoolSize = {};
	descPoolSize.type = vk::DescriptorType::eStorageBuffer;
	descPoolSize.descriptorCount = 2; // デスクリプタに結び付けるバッファの数.

	vk::DescriptorPoolCreateInfo descPoolInfo = {};
	descPoolInfo.poolSizeCount = 1;
	descPoolInfo.pPoolSizes = &descPoolSize;
	descPoolInfo.maxSets = 1;
	descPoolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	vk::UniqueDescriptorPool descPool = m_device->createDescriptorPoolUnique(descPoolInfo);

	// デスクリプタセットレイアウトの作成.
	std::vector<vk::DescriptorSetLayoutBinding> descSetLayoutBinding(2);
	descSetLayoutBinding[0].binding = 0;
	descSetLayoutBinding[0].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[0].descriptorCount = 1;
	descSetLayoutBinding[0].stageFlags = vk::ShaderStageFlagBits::eCompute;

	descSetLayoutBinding[1].binding = 1;
	descSetLayoutBinding[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[1].descriptorCount = 1;
	descSetLayoutBinding[1].stageFlags = vk::ShaderStageFlagBits::eCompute;

	vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo = {};
	descSetLayoutInfo.bindingCount = static_cast<uint32_t> (descSetLayoutBinding.size());
	descSetLayoutInfo.pBindings = descSetLayoutBinding.data();
	vk::UniqueDescriptorSetLayout descSetLayout = m_device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

	// デスクリプタセットの作成.
	vk::DescriptorSetAllocateInfo descSetInfo = {};
	descSetInfo.descriptorPool = descPool.get();
	descSetInfo.descriptorSetCount = 1;
	descSetInfo.pSetLayouts = &descSetLayout.get();

	vk::UniqueDescriptorSet descSet = std::move(m_device->allocateDescriptorSetsUnique(descSetInfo)[0]);

	// コンピュートパイプラインの作成.
	vk::PushConstantRange pushConstantRange = {};
	pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(int) * 2 + sizeof(float);

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descSetLayout.get();
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

	vk::UniquePipelineLayout pipelineLayout = m_device->createPipelineLayoutUnique(pipelineLayoutInfo);

	vk::ComputePipelineCreateInfo pipelineInfo = {};
	pipelineInfo.stage.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
	pipelineInfo.stage.stage = vk::ShaderStageFlagBits::eCompute;
	pipelineInfo.stage.module = computeShaderModule.get();
	pipelineInfo.stage.pName = "main";
	pipelineInfo.layout = pipelineLayout.get();

	vk::UniquePipelineCache pipelineCache = m_device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo());

	auto wrapped = m_device->createComputePipelinesUnique(pipelineCache.get(), pipelineInfo);
	vk::UniquePipeline computePipeline = std::move(wrapped.value[0]);

	// デスクリプタにbind.
	vk::DescriptorBufferInfo bufferInfoA = {};
	bufferInfoA.buffer = bufferA.get();
	bufferInfoA.offset = 0;
	bufferInfoA.range = bufferSizeInA;

	vk::DescriptorBufferInfo bufferInfoC = {};
	bufferInfoC.buffer = bufferC.get();
	bufferInfoC.offset = 0;
	bufferInfoC.range = bufferSizeOut;

	std::vector<vk::WriteDescriptorSet> descWrites(2);
	descWrites[0].dstSet = descSet.get();
	descWrites[0].dstBinding = 0;
	descWrites[0].dstArrayElement = 0;
	descWrites[0].descriptorCount = 1;
	descWrites[0].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[0].pBufferInfo = &bufferInfoA;


	descWrites[1].dstSet = descSet.get();
	descWrites[1].dstBinding = 1;
	descWrites[1].dstArrayElement = 0;
	descWrites[1].descriptorCount = 1;
	descWrites[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[1].pBufferInfo = &bufferInfoC;

	m_device->updateDescriptorSets(descWrites, nullptr);

	vk::CommandBufferAllocateInfo cmdAllocateInfo = {};
	cmdAllocateInfo.level = vk::CommandBufferLevel::ePrimary;
	cmdAllocateInfo.commandPool = m_commandPool.get();
	cmdAllocateInfo.commandBufferCount = 1;

	vk::UniqueCommandBuffer calcCommandBuffer = std::move(m_device->allocateCommandBuffersUnique(cmdAllocateInfo)[0]);

	vk::CommandBufferBeginInfo beginInfo = {};
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	calcCommandBuffer->begin(beginInfo);
	calcCommandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline.get());
	calcCommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descSet.get(), {});

	// プッシュコンスタントの設定.
	struct pushdata {
		int width;
		int height;
		float value;
	} constantValue;

	constantValue.width = width;
	constantValue.height = height;
	constantValue.value = value;


	calcCommandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0,
		sizeof(constantValue), &constantValue);

	calcCommandBuffer->dispatch((width + 15) / 16, (height + 15) / 16, 1);
	calcCommandBuffer->end();

	const vk::PipelineStageFlags stageCalc = vk::PipelineStageFlagBits::eComputeShader;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &calcCommandBuffer.get();
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.setPWaitSemaphores(&m_semaphore.get());
	submitInfo.pWaitDstStageMask = &stageCalc;

	m_queue.submit(submitInfo);

	data = m_device->mapMemory(stagingBufferMemoryC.get(), 0, bufferSizeOut);
	vk::UniqueCommandBuffer getCommandBuffer = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		bufferC.get(), stagingBufferC.get(), bufferSizeOut, vk::CommandBufferLevel::ePrimary);

	const vk::PipelineStageFlags stageTrans = vk::PipelineStageFlagBits::eTransfer;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &getCommandBuffer.get();
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.setPWaitSemaphores(&m_semaphore.get());
	submitInfo.pWaitDstStageMask = &stageTrans;

	m_queue.submit(submitInfo, m_fence.get());

	const std::vector< vk::Fence > fences{ m_fence.get() };
	if (m_device->waitForFences(fences, true, 10000000000u) != vk::Result::eSuccess) {
		abort();
	}

	memcpy(dataC.data(), data, bufferSizeOut);
	m_device->unmapMemory(stagingBufferMemoryC.get());


	for (int i = 0; i < height; ++i) {
		std::copy(dataC.begin() + i * width, dataC.begin() + (i + 1) * width, ans[i].begin());
	}

	return ans;
}

std::vector<std::vector<float>> Matrix::diff(const std::vector<std::vector<float>>& matA, const std::vector<std::vector<float>>& matB) {
	if (matA.empty() || matB.empty()) {
		throw std::runtime_error("matrix is empty.");
		abort();
	}

	int heightA = (int)(matA.size());
	int widthA = (int)(matA[0].size());
	int heightB = (int)(matB.size());
	int widthB = (int)(matB[0].size());

	if ((widthA != widthB) || (heightA != heightB)) {
		throw std::runtime_error("don't match matrix size for calculation.");
		abort();
	}

	int width = widthA;
	int height = heightA;

	// バッファの作成.
	std::vector<std::vector<float>> ans(height, std::vector<float>(width));

	std::vector<float> dataA(width * height);
	std::vector<float> dataB(width * height);
	std::vector<float> dataC(width * height);

	for (int i = 0; i < height; ++i) {
		std::copy(matA[i].begin(), matA[i].end(), dataA.begin() + i * width);
	}

	for (int i = 0; i < height; ++i) {
		std::copy(matB[i].begin(), matB[i].end(), dataB.begin() + i * width);
	}

	vk::DeviceSize bufferSizeInA = sizeof(float) * width * height;
	vk::DeviceSize bufferSizeInB = sizeof(float) * width * height;
	vk::DeviceSize bufferSizeOut = sizeof(float) * height * width;

	vk::UniqueDeviceMemory stagingBufferMemoryA, stagingBufferMemoryB, stagingBufferMemoryC, bufferAMemory, bufferBMemory, bufferCMemory;

	vk::UniqueBuffer stagingBufferA = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInA,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryA);

	vk::UniqueBuffer stagingBufferB = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInB,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryB);

	vk::UniqueBuffer stagingBufferC = createBuffer(m_device.get(), m_physicalDevice, bufferSizeOut,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBufferMemoryC);

	vk::UniqueBuffer bufferA = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInA,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferAMemory);

	vk::UniqueBuffer bufferB = createBuffer(m_device.get(), m_physicalDevice, bufferSizeInB,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferBMemory);

	vk::UniqueBuffer bufferC = createBuffer(m_device.get(), m_physicalDevice, bufferSizeOut,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal, bufferCMemory);

	// コマンドバッファの作成.
	void* data;
	data = m_device->mapMemory(stagingBufferMemoryA.get(), 0, bufferSizeInA);
	std::memcpy(data, dataA.data(), bufferSizeInA);
	vk::UniqueCommandBuffer commandBufferA = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		stagingBufferA.get(), bufferA.get(), bufferSizeInA, vk::CommandBufferLevel::ePrimary);
	m_device->unmapMemory(stagingBufferMemoryA.get());

	data = m_device->mapMemory(stagingBufferMemoryB.get(), 0, bufferSizeInB);
	std::memcpy(data, dataB.data(), bufferSizeInB);
	vk::UniqueCommandBuffer commandBufferB = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		stagingBufferB.get(), bufferB.get(), bufferSizeInB, vk::CommandBufferLevel::ePrimary);
	m_device->unmapMemory(stagingBufferMemoryB.get());

	std::vector<vk::CommandBuffer> commandBuffers = { commandBufferA.get(), commandBufferB.get() };

	// キューに流して実行.
	vk::SubmitInfo submitInfo = {};
	submitInfo.commandBufferCount = static_cast<uint32_t> (commandBuffers.size());
	submitInfo.pCommandBuffers = commandBuffers.data();
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.setPSignalSemaphores(&m_semaphore.get());
	m_queue.submit(submitInfo);

	// シェーダモジュールの設定. 修正必要.
	std::vector<char> code = readFile("src/matDiff.spv");
	vk::UniqueShaderModule computeShaderModule = createShaderModule(m_device.get(), code);

	// デスクリプタプールの作成.
	vk::DescriptorPoolSize descPoolSize = {};
	descPoolSize.type = vk::DescriptorType::eStorageBuffer;
	descPoolSize.descriptorCount = 3; // デスクリプタに結び付けるバッファの数.

	vk::DescriptorPoolCreateInfo descPoolInfo = {};
	descPoolInfo.poolSizeCount = 1;
	descPoolInfo.pPoolSizes = &descPoolSize;
	descPoolInfo.maxSets = 1;
	descPoolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	vk::UniqueDescriptorPool descPool = m_device->createDescriptorPoolUnique(descPoolInfo);

	// デスクリプタセットレイアウトの作成.
	std::vector<vk::DescriptorSetLayoutBinding> descSetLayoutBinding(3);
	descSetLayoutBinding[0].binding = 0;
	descSetLayoutBinding[0].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[0].descriptorCount = 1;
	descSetLayoutBinding[0].stageFlags = vk::ShaderStageFlagBits::eCompute;

	descSetLayoutBinding[1].binding = 1;
	descSetLayoutBinding[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[1].descriptorCount = 1;
	descSetLayoutBinding[1].stageFlags = vk::ShaderStageFlagBits::eCompute;

	descSetLayoutBinding[2].binding = 2;
	descSetLayoutBinding[2].descriptorType = vk::DescriptorType::eStorageBuffer;
	descSetLayoutBinding[2].descriptorCount = 1;
	descSetLayoutBinding[2].stageFlags = vk::ShaderStageFlagBits::eCompute;

	vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo = {};
	descSetLayoutInfo.bindingCount = static_cast<uint32_t> (descSetLayoutBinding.size());
	descSetLayoutInfo.pBindings = descSetLayoutBinding.data();
	vk::UniqueDescriptorSetLayout descSetLayout = m_device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

	// デスクリプタセットの作成.
	vk::DescriptorSetAllocateInfo descSetInfo = {};
	descSetInfo.descriptorPool = descPool.get();
	descSetInfo.descriptorSetCount = 1;
	descSetInfo.pSetLayouts = &descSetLayout.get();

	vk::UniqueDescriptorSet descSet = std::move(m_device->allocateDescriptorSetsUnique(descSetInfo)[0]);

	// コンピュートパイプラインの作成.
	vk::PushConstantRange pushConstantRange = {};
	pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(int) * 2;

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descSetLayout.get();
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

	vk::UniquePipelineLayout pipelineLayout = m_device->createPipelineLayoutUnique(pipelineLayoutInfo);

	vk::ComputePipelineCreateInfo pipelineInfo = {};
	pipelineInfo.stage.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
	pipelineInfo.stage.stage = vk::ShaderStageFlagBits::eCompute;
	pipelineInfo.stage.module = computeShaderModule.get();
	pipelineInfo.stage.pName = "main";
	pipelineInfo.layout = pipelineLayout.get();

	vk::UniquePipelineCache pipelineCache = m_device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo());

	auto wrapped = m_device->createComputePipelinesUnique(pipelineCache.get(), pipelineInfo);
	vk::UniquePipeline computePipeline = std::move(wrapped.value[0]);

	// デスクリプタにbind.
	vk::DescriptorBufferInfo bufferInfoA = {};
	bufferInfoA.buffer = bufferA.get();
	bufferInfoA.offset = 0;
	bufferInfoA.range = bufferSizeInA;

	vk::DescriptorBufferInfo bufferInfoB = {};
	bufferInfoB.buffer = bufferB.get();
	bufferInfoB.offset = 0;
	bufferInfoB.range = bufferSizeInB;

	vk::DescriptorBufferInfo bufferInfoC = {};
	bufferInfoC.buffer = bufferC.get();
	bufferInfoC.offset = 0;
	bufferInfoC.range = bufferSizeOut;

	std::vector<vk::WriteDescriptorSet> descWrites(3);
	descWrites[0].dstSet = descSet.get();
	descWrites[0].dstBinding = 0;
	descWrites[0].dstArrayElement = 0;
	descWrites[0].descriptorCount = 1;
	descWrites[0].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[0].pBufferInfo = &bufferInfoA;


	descWrites[1].dstSet = descSet.get();
	descWrites[1].dstBinding = 1;
	descWrites[1].dstArrayElement = 0;
	descWrites[1].descriptorCount = 1;
	descWrites[1].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[1].pBufferInfo = &bufferInfoB;

	descWrites[2].dstSet = descSet.get();
	descWrites[2].dstBinding = 2;
	descWrites[2].dstArrayElement = 0;
	descWrites[2].descriptorCount = 1;
	descWrites[2].descriptorType = vk::DescriptorType::eStorageBuffer;
	descWrites[2].pBufferInfo = &bufferInfoC;

	m_device->updateDescriptorSets(descWrites, nullptr);

	vk::CommandBufferAllocateInfo cmdAllocateInfo = {};
	cmdAllocateInfo.level = vk::CommandBufferLevel::ePrimary;
	cmdAllocateInfo.commandPool = m_commandPool.get();
	cmdAllocateInfo.commandBufferCount = 1;

	vk::UniqueCommandBuffer calcCommandBuffer = std::move(m_device->allocateCommandBuffersUnique(cmdAllocateInfo)[0]);

	vk::CommandBufferBeginInfo beginInfo = {};
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	calcCommandBuffer->begin(beginInfo);
	calcCommandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline.get());
	calcCommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout.get(), 0, descSet.get(), {});

	// プッシュコンスタントの設定.
	struct pushdata {
		int width;
		int height;
	} constantValue;

	constantValue.width = width;
	constantValue.height = height;


	calcCommandBuffer->pushConstants(pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0,
		sizeof(constantValue), &constantValue);

	calcCommandBuffer->dispatch((width + 15) / 16, (height + 15) / 16, 1);
	calcCommandBuffer->end();

	const vk::PipelineStageFlags stageCalc = vk::PipelineStageFlagBits::eComputeShader;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &calcCommandBuffer.get();
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.setPWaitSemaphores(&m_semaphore.get());
	submitInfo.pWaitDstStageMask = &stageCalc;

	m_queue.submit(submitInfo);

	data = m_device->mapMemory(stagingBufferMemoryC.get(), 0, bufferSizeOut);
	vk::UniqueCommandBuffer getCommandBuffer = makeCopyCommandBuffer(m_device.get(), m_commandPool.get(),
		bufferC.get(), stagingBufferC.get(), bufferSizeOut, vk::CommandBufferLevel::ePrimary);

	const vk::PipelineStageFlags stageTrans = vk::PipelineStageFlagBits::eTransfer;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &getCommandBuffer.get();
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.setPWaitSemaphores(&m_semaphore.get());
	submitInfo.pWaitDstStageMask = &stageTrans;

	m_queue.submit(submitInfo, m_fence.get());

	const std::vector< vk::Fence > fences{ m_fence.get() };
	if (m_device->waitForFences(fences, true, 10000000000u) != vk::Result::eSuccess) {
		abort();
	}

	memcpy(dataC.data(), data, bufferSizeOut);
	m_device->unmapMemory(stagingBufferMemoryC.get());


	for (int i = 0; i < height; ++i) {
		std::copy(dataC.begin() + i * width, dataC.begin() + (i + 1) * width, ans[i].begin());
	}

	return ans;
}