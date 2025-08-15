"use client"

import { useState, useEffect } from "react"
import { ChatMessage } from "@/components/chat-message"
import { ChatInput } from "@/components/chat-input"
import { Sidebar } from "@/components/sidebar"
import { Button } from "@/components/ui/button"
import { Menu, Plus } from "lucide-react"
import { saveChatsToStorage, loadChatsFromStorage } from "@/lib/storage"

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

export default function HomePage() {
  const [chats, setChats] = useState<Chat[]>([])
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [documents, setDocuments] = useState<Document[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [isMessageLoading, setIsMessageLoading] = useState(false)

  useEffect(() => {
    const loadedChats = loadChatsFromStorage()
    setChats(loadedChats)
    setIsLoading(false)
  }, [])

  useEffect(() => {
    if (!isLoading) {
      saveChatsToStorage(chats)
    }
  }, [chats, isLoading])

  const currentChat = chats.find((chat) => chat.id === currentChatId)

  const createNewChat = () => {
    const newChat: Chat = {
      id: Date.now().toString(),
      title: "New Chat",
      messages: [],
      createdAt: new Date(),
    }
    setChats((prev) => [newChat, ...prev])
    setCurrentChatId(newChat.id)
  }

  const sendMessage = async (content: string) => {
    if (!currentChatId) {
      createNewChat()
      return
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      role: "user",
      timestamp: new Date(),
    }

    setChats((prev) =>
      prev.map((chat) =>
        chat.id === currentChatId
          ? {
              ...chat,
              messages: [...chat.messages, userMessage],
              title: chat.messages.length === 0 ? content.slice(0, 30) + "..." : chat.title,
            }
          : chat,
      ),
    )

    setIsMessageLoading(true)

    try {
      const currentChat = chats.find((chat) => chat.id === currentChatId)
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: content,
          chatId: currentChatId,
          history: currentChat?.messages || [],
          documents: documents.map((doc) => ({
            id: doc.id,
            name: doc.name,
            content: doc.content,
          })),
        }),
      })

      const data = await response.json()

      if (data.success) {
        setChats((prev) =>
          prev.map((chat) =>
            chat.id === currentChatId
              ? {
                  ...chat,
                  messages: [...chat.messages, data.message],
                }
              : chat,
          ),
        )
      } else {
        console.error("API Error:", data.error)
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: "Sorry, I'm having trouble responding right now. Please try again.",
          role: "assistant",
          timestamp: new Date(),
        }

        setChats((prev) =>
          prev.map((chat) =>
            chat.id === currentChatId
              ? {
                  ...chat,
                  messages: [...chat.messages, errorMessage],
                }
              : chat,
          ),
        )
      }
    } catch (error) {
      console.error("Network error:", error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "Network error. Please check your connection and try again.",
        role: "assistant",
        timestamp: new Date(),
      }

      setChats((prev) =>
        prev.map((chat) =>
          chat.id === currentChatId
            ? {
                ...chat,
                messages: [...chat.messages, errorMessage],
              }
            : chat,
        ),
      )
    } finally {
      setIsMessageLoading(false)
    }
  }

  const selectChat = (chatId: string) => {
    setCurrentChatId(chatId)
    setSidebarOpen(false)
  }

  const deleteChat = (chatId: string) => {
    setChats((prev) => prev.filter((chat) => chat.id !== chatId))
    if (currentChatId === chatId) {
      setCurrentChatId(null)
    }
  }

  const handleDocumentUpload = (document: Document) => {
    setDocuments((prev) => [document, ...prev])
  }

  const handleDocumentDelete = (documentId: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.id !== documentId))
  }

  if (isLoading) {
    return (
      <div className="flex h-screen bg-background items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading your chats...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <Sidebar
        chats={chats}
        currentChatId={currentChatId}
        onSelectChat={selectChat}
        onDeleteChat={deleteChat}
        onNewChat={createNewChat}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        documents={documents}
        onDocumentUpload={handleDocumentUpload}
        onDocumentDelete={handleDocumentDelete}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(true)} className="md:hidden">
              <Menu className="h-5 w-5" />
            </Button>
            <h1 className="text-xl font-semibold">AI Assistant</h1>
          </div>
          <Button onClick={createNewChat} size="sm" className="flex items-center gap-2">
            <Plus className="h-4 w-4" />
            New Chat
          </Button>
        </header>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4">
          {currentChat?.messages.length === 0 || !currentChat ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <h2 className="text-2xl font-semibold mb-2">Welcome to AI Assistant</h2>
                <p className="text-muted-foreground mb-4">Start a conversation by typing a message below</p>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-4">
              {currentChat.messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
            </div>
          )}
        </div>

        {/* Chat Input */}
        <div className="border-t border-border p-4">
          <div className="w-full mx-auto">
            <ChatInput onSendMessage={sendMessage} disabled={isMessageLoading} />
            {isMessageLoading && (
              <div className="flex items-center justify-center mt-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary mr-2"></div>
                <span className="text-sm text-muted-foreground">AI is thinking...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
