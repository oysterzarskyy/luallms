--[[
    luallms - main wrapper
    github: oystah/luallms
]]

--!strict
local Brain = require(script.brain)
local AgentSearch = require(script.agentsearch) -- Requiring the renamed module file

local LuaLLMs = {}

-- expose constructors to user scripts
LuaLLMs.CreateBrain = Brain.new
LuaLLMs.LoadSearchAgent = AgentSearch.new 

print("[luallms] initialized, running lullzm-coder:mini.gentle")
return LuaLLMs
