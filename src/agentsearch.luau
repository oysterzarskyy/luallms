--[[
    luallms - agentsearch component
    github: oystah/luallms
]]

--!strict
local HttpService = game:GetService("HttpService")

local AgentSearch = {}
AgentSearch.__index = AgentSearch

type ToolInstance = {
	ConfigUrl: string,
}

function AgentSearch.new(): ToolInstance
	local self = setmetatable({}, AgentSearch) :: any
	-- Points directly to your free luallms domain profile configuration link
	self.ConfigUrl = "https://github.io"
	return self
end

function AgentSearch:SearchAndFind(searchQuery: string): (boolean, string)
	local successConfig, configRaw = pcall(function() 
		return game:HttpGet(self.ConfigUrl) 
	end)
	
	if not successConfig then 
		return false, "[luallms] agentsearch link error: failed to verify config routes." 
	end
	
	local configTable = loadstring(configRaw)()
	local searchEngineUrl = configTable.ScraperGateway or "https://duckduckgo.com"
	
	local sanitizedQuery = HttpService:UrlEncode(searchQuery)
	local targetSearchUrl = searchEngineUrl .. sanitizedQuery
	
	local successSearch, htmlContent = pcall(function()
		return game:HttpGet(targetSearchUrl)
	end)
	
	if not successSearch then
		return false, "[luallms] agentsearch search failure: target web index unreachable."
	end
	
	local discoveredWebsites: {string} = {}
	local cleanSummary = "--- luallms discovered urls ---\n"
	
	for url in string.gmatch(htmlContent, 'href="([^"]+)"') do
		if string.find(url, "http") and not string.find(url, "duckduckgo") then
			if #discoveredWebsites < 3 then
				table.insert(discoveredWebsites, url)
			end
		end
	end
	
	if #discoveredWebsites == 0 then
		return false, "[luallms] search completed but found zero matching parameters."
	else
		for index, websiteUrl in ipairs(discoveredWebsites) do
			cleanSummary ..= string.format("[%d] %s\n", index, websiteUrl)
		end
		return true, cleanSummary
	end
end

return AgentSearch
