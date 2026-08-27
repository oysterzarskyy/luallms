--[[
    lullmz brain thing - i wrote this because standard neural nets are too slow
    pray that Luau native codegen actually speeds this up lmao
    - oystah
]]

--!strict
--!native

local Brain = {}
Brain.__index = Brain

-- types for the strict compiler so it stops yelling at me
export type Matrix = { {number} }
export type LayerStructure = { number }

type BrainInstance = {
	structure: LayerStructure,
	weights: { Matrix },
	biases: { {number} },
	learningRate: number,
}

-- generic sigmoid, good enough for now
local function sigmoid(x: number): number
	return 1 / (1 + math.exp(-x))
end

-- gradient helper for backprop
local function sigmoidDerivative(y: number): number
	return y * (1 - y)
end

-- constructor
function Brain.new(structure: LayerStructure, learningRate: number?): BrainInstance
	local self = setmetatable({}, Brain) :: any
	
	self.structure = structure
	self.learningRate = learningRate or 0.1
	self.weights = {}
	self.biases = {}
	
	-- slap random small numbers everywhere to initialize
	for i = 1, #structure - 1 do
		local rows = structure[i + 1]
		local cols = structure[i]
		
		local layerWeights: Matrix = {}
		local layerBiases: {number} = {}
		
		for r = 1, rows do
			layerWeights[r] = {}
			for c = 1, cols do
				-- range -1 to 1, hopefully avoids exploding gradients
				layerWeights[r][c] = math.random() * 2 - 1 
			end
			layerBiases[r] = math.random() * 2 - 1
		end
		
		table.insert(self.weights, layerWeights)
		table.insert(self.biases, layerBiases)
	end
	
	return self
end

-- feedforward loop
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

-- the agonizing math part
function Brain:Train(inputs: {number}, targets: {number})
	-- 1. copy-pasted the forward pass here because we need history for backprop
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
	
	-- 2. find out how wrong the prediction was
	local currentErrors: {number} = {}
	local finalOutputs = layerOutputs[#layerOutputs]
	
	for i = 1, #targets do
		currentErrors[i] = targets[i] - finalOutputs[i]
	end
	
	-- loop backwards through layers, don't mess up the indices
	for layerIdx = #self.weights, 1, -1 do
		local layerWeights = self.weights[layerIdx]
		local outputs = layerOutputs[layerIdx + 1]
		local prevOutputs = layerOutputs[layerIdx]
		
		local nextErrors: {number} = {}
		for c = 1, #layerWeights do
			nextErrors[c] = 0
		end
		
		for r = 1, #layerWeights do
			-- actual calculus. do not touch or everything breaks
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
