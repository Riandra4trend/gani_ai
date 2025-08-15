"use client"

import { useState, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DocumentUpload } from "@/components/document-upload"
import { SearchHighlight } from "@/components/search-highlight"
import { Plus, MessageSquare, MoreVertical, Trash2, X, FileText, Search, Clock } from "lucide-react"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { searchChats, getSearchSuggestions } from "@/lib/search"

interface Message {
  id: string
  content: string
  role: "user" | "assistant"
  timestamp: Date
}

interface Chat {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
}

interface Document {
  id: string
  name: string
  size: number
  type: string
  content: string
  uploadedAt: Date
}

interface SidebarProps {
  chats: Chat[]
  currentChatId: string | null
  onSelectChat: (chatId: string) => void
  onDeleteChat: (chatId: string) => void
  onNewChat: () => void
  isOpen: boolean
  onClose: () => void
  documents: Document[]
  onDocumentUpload: (document: Document) => void
  onDocumentDelete: (documentId: string) => void
  searchQuery: string
  onSearchChange: (query: string) => void
}

export function Sidebar({
  chats,
  currentChatId,
  onSelectChat,
  onDeleteChat,
  onNewChat,
  isOpen,
  onClose,
  documents,
  onDocumentUpload,
  onDocumentDelete,
  searchQuery,
  onSearchChange,
}: SidebarProps) {
  const [showSuggestions, setShowSuggestions] = useState(false)

  const searchResults = useMemo(() => {
    if (!searchQuery.trim()) return []
    return searchChats(chats, searchQuery)
  }, [chats, searchQuery])

  const suggestions = useMemo(() => {
    if (!searchQuery.trim() || searchQuery.length < 2) return []
    return getSearchSuggestions(chats, searchQuery)
  }, [chats, searchQuery])

  const displayChats = searchQuery.trim() ? searchResults.map((result) => result.chat) : chats

  const handleSearchFocus = () => {
    setShowSuggestions(true)
  }

  const handleSearchBlur = () => {
    // Delay hiding suggestions to allow clicking on them
    setTimeout(() => setShowSuggestions(false), 200)
  }

  const handleSuggestionClick = (suggestion: string) => {
    onSearchChange(suggestion)
    setShowSuggestions(false)
  }

  const clearSearch = () => {
    onSearchChange("")
    setShowSuggestions(false)
  }

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && <div className="fixed inset-0 bg-black/50 z-40 md:hidden" onClick={onClose} />}

      {/* Sidebar */}
      <div
        className={cn(
          "fixed left-0 top-0 z-50 h-full w-80 bg-sidebar border-r border-sidebar-border transform transition-transform duration-200 ease-in-out md:relative md:translate-x-0",
          isOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
            <h2 className="font-semibold text-sidebar-foreground">AI Assistant</h2>
            <div className="flex items-center gap-2">
              <Button
                onClick={onNewChat}
                size="sm"
                variant="ghost"
                className="text-sidebar-foreground hover:bg-sidebar-accent"
              >
                <Plus className="h-4 w-4" />
              </Button>
              <Button
                onClick={onClose}
                size="sm"
                variant="ghost"
                className="md:hidden text-sidebar-foreground hover:bg-sidebar-accent"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <Tabs defaultValue="chats" className="flex-1 flex flex-col">
          <TabsList className="grid grid-cols-2 w-[300px] m-2 overflow-hidden">
            <TabsTrigger 
              value="chats" 
              className="flex items-center gap-1 truncate px-1 py-1"
            >
              <MessageSquare className="h-4 w-2 shrink-0" />
              <span className="truncate">Chats</span>
            </TabsTrigger>
            <TabsTrigger 
              value="documents" 
              className="flex items-center gap-1 truncate px-1 py-1"
            >
              <FileText className="h-4 w-2 shrink-0" />
              <span className="truncate">Docs</span>
            </TabsTrigger>
          </TabsList>

            <TabsContent value="chats" className="flex-1 flex flex-col m-0">
              <div className="p-2 relative">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search chats..."
                    value={searchQuery}
                    onChange={(e) => onSearchChange(e.target.value)}
                    onFocus={handleSearchFocus}
                    onBlur={handleSearchBlur}
                    className="pl-9 pr-8"
                  />
                  {searchQuery && (
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={clearSearch}
                      className="absolute right-1 top-1/2 transform -translate-y-1/2 h-6 w-6 text-muted-foreground hover:text-foreground"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  )}
                </div>

                {/* Search Suggestions */}
                {showSuggestions && suggestions.length > 0 && (
                  <Card className="absolute top-full left-2 right-2 z-10 mt-1 p-2 shadow-lg">
                    <div className="space-y-1">
                      {suggestions.map((suggestion, index) => (
                        <Button
                          key={index}
                          variant="ghost"
                          size="sm"
                          onClick={() => handleSuggestionClick(suggestion)}
                          className="w-full justify-start text-left h-8"
                        >
                          <Clock className="h-3 w-3 mr-2 text-muted-foreground" />
                          {suggestion}
                        </Button>
                      ))}
                    </div>
                  </Card>
                )}
              </div>

              {/* Search Results Info */}
              {searchQuery && (
                <div className="px-2 pb-2">
                  <p className="text-xs text-muted-foreground">
                    {searchResults.length} result{searchResults.length !== 1 ? "s" : ""} for "{searchQuery}"
                  </p>
                </div>
              )}

              {/* Chat List */}
              <ScrollArea className="flex-1 p-2">
                {displayChats.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-32 text-center">
                    <MessageSquare className="h-8 w-8 text-sidebar-foreground/50 mb-2" />
                    <p className="text-sm text-sidebar-foreground/70">
                      {searchQuery ? "No chats found" : "No chats yet"}
                    </p>
                    <p className="text-xs text-sidebar-foreground/50">
                      {searchQuery ? "Try a different search" : "Start a new conversation"}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-1">
                    {displayChats.map((chat) => {
                      const searchResult = searchResults.find((result) => result.chat.id === chat.id)
                      return (
                        <Card
                          key={chat.id}
                          className={cn(
                            "p-3 cursor-pointer hover:bg-sidebar-accent transition-colors group",
                            currentChatId === chat.id && "bg-sidebar-accent border-sidebar-primary",
                          )}
                          onClick={() => onSelectChat(chat.id)}
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0">
                              <h3 className="font-medium text-sm text-sidebar-foreground truncate">
                                <SearchHighlight text={chat.title} searchQuery={searchQuery} />
                              </h3>
                              {searchResult?.matchedMessage && (
                                <p className="text-xs text-sidebar-foreground/70 mt-1 line-clamp-2">
                                  <SearchHighlight
                                    text={searchResult.matchedMessage.content}
                                    searchQuery={searchQuery}
                                  />
                                </p>
                              )}
                              <div className="flex items-center gap-2 mt-1">
                                <p className="text-xs text-sidebar-foreground/70">{chat.messages.length} messages</p>
                                <p className="text-xs text-sidebar-foreground/50">
                                  {chat.createdAt.toLocaleDateString()}
                                </p>
                              </div>
                            </div>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-6 w-6 opacity-0 group-hover:opacity-100 text-sidebar-foreground hover:bg-sidebar-accent"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <MoreVertical className="h-3 w-3" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    onDeleteChat(chat.id)
                                  }}
                                  className="text-destructive focus:text-destructive"
                                >
                                  <Trash2 className="h-4 w-4 mr-2" />
                                  Delete
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </div>
                        </Card>
                      )
                    })}
                  </div>
                )}
              </ScrollArea>
            </TabsContent>

            <TabsContent value="documents" className="flex-1 flex flex-col m-0">
              <ScrollArea className="flex-1 p-2">
                <DocumentUpload
                  documents={documents}
                  onDocumentUpload={onDocumentUpload}
                  onDocumentDelete={onDocumentDelete}
                />
              </ScrollArea>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </>
  )
}
