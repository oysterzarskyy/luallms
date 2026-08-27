--[[
    luallms - main wrapper
    github: oystah/luallms
]]

--!strict
local Brain = require(script.brain)
local Agsetol = require(script.agsetol) 

local LuaLLMs = {}

-- expose constructors to user scripts
LuaLLMs.CreateBrain = Brain.new
LuaLLMs.LoadSearchAgent = Agsetol.new 

print("[luallms] initialized")
return LuaLLMs
