--[[
    luallms - brain engine logic
    github: oystah/luallms
]]

--!strict
--!native

local Brain = {}
Brain.__index = Brain

export type Matrix = { {number} }
export type LayerStructure = { number }

type BrainInstance = {
	structure: LayerStructure,
	weights: { Matrix },
	biases: { {number} },
	learningRate: number,
}

local function sigmoid(x: number): number
	return 1 / (1 + math.exp(-x))
end

local function sigmoidDerivative(y: number): number
	return y * (1 - y)
end

function Brain.new(structure: LayerStructure, learningRate: number?): BrainInstance
	local self = setmetatable({}, Brain) :: any
	
	self.structure = structure
	self.learningRate = learningRate or 0.1
	self.weights = {}
	self.biases = {}
	
	for i = 1, #structure - 1 do
		local rows = structure[i + 1]
		local cols = structure[i]
		
		local layerWeights: Matrix = {}
		local layerBiases: {number} = {}
		
		for r = 1, rows do
			layerWeights[r] = {}
			for c = 1, cols do
				layerWeights[r][c] = math.random() * 2 - 1
			end
			layerBiases[r] = math.random() * 2 - 1
		end
		
		table.insert(self.weights, layerWeights)
		table.insert(self.biases, layerBiases)
	end
	
	return self
end

function Brain:Forward(inputs: {number}): {number}
	local currentOutputs = table.clone(inputs)
	
	for layerIdx, layerWeights in ipairs(self.weights) do
		local nextOutputs: {number} = {}
		local layerBiases = self.biases[layerIdx]
		
		for r = 1, #layerWeights do
			local sum = layerBiases[r]
			for c = 1, #layerWeights[r] do
				sum += currentOutputs[c] * layerWeights[r][c]
			end
			nextOutputs[r] = sigmoid(sum)
		end
		currentOutputs = nextOutputs
	end
	
	return currentOutputs
end

function Brain:Train(inputs: {number}, targets: {number})
	local layerOutputs: { {number} } = { table.clone(inputs) }
	local currentOutputs = table.clone(inputs)
	
	for layerIdx, layerWeights in ipairs(self.weights) do
		local nextOutputs: {number} = {}
		local layerBiases = self.biases[layerIdx]
		
		for r = 1, #layerWeights do
			local sum = layerBiases[r]
			for c = 1, #layerWeights[r] do
				sum += currentOutputs[c] * layerWeights[r][c]
			end
			nextOutputs[r] = sigmoid(sum)
		end
		currentOutputs = nextOutputs
		table.insert(layerOutputs, table.clone(currentOutputs))
	end
	
	local currentErrors: {number} = {}
	local finalOutputs = layerOutputs[#layerOutputs]
	
	for i = 1, #targets do
		currentErrors[i] = targets[i] - finalOutputs[i]
	end
	
	for layerIdx = #self.weights, 1, -1 do
		local layerWeights = self.weights[layerIdx]
		local outputs = layerOutputs[layerIdx + 1]
		local prevOutputs = layerOutputs[layerIdx]
		
		local nextErrors: {number} = {}
		for c = 1, #layerWeights do
			nextErrors[c] = 0
		end
		
		for r = 1, #layerWeights do
			local gradient = currentErrors[r] * sigmoidDerivative(outputs[r]) * self.learningRate
			self.biases[layerIdx][r] += gradient
			
			for c = 1, #layerWeights[r] do
				nextErrors[c] += layerWeights[r][c] * currentErrors[r]
				self.weights[layerIdx][r][c] += gradient * prevOutputs[c]
			end
		end
		currentErrors = nextErrors
	end
end

return Brain
