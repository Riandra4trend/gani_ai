"use client"

import { useState, useEffect, useCallback } from "react"
import { ChatMessage } from "@/components/chat-message"
import { ChatInput } from "@/components/chat-input"
import { Sidebar } from "@/components/sidebar"
import { DocumentUpload } from "@/components/document-upload"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Menu, Plus, Brain, Clock, TrendingUp, Upload, FileText, Activity } from "lucide-react"
import { saveChatsToStorage, loadChatsFromStorage } from "@/lib/storage"

interface Message {
  id: string
  content: string
  role: "user" | "assistant"
  timestamp: Date
  sources?: Array<{
    content: string
    metadata: Record<string, any>
    score: number
    source: string
  }>
  processing_time?: number
}

interface Chat {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  session_id?: string
}

interface Document {
  id: string
  name: string
  size: number
  type: string
  content: string
  uploadedAt: Date
  status?: "processing" | "completed" | "error"
}

interface ChatResponse {
  session_id: string
  response: string
  sources?: Array<{
    content: string
    metadata: Record<string, any>
    score: number
    source: string
  }>
  timestamp: string
  processing_time: number
}

interface DocumentUploadResponse {
  document_id: string
  filename: string
  status: string
  message: string
  chunks_created?: number
}

interface AppMetrics {
  total_chats: number
  total_documents: number
  successful_responses: number
  failed_responses: number
  uptime_seconds: number
  documents_processed: number
}

interface ChatHistoryResponse {
  session_id: string
  messages: Array<{
    role: string
    content: string
    timestamp: string
  }>
  created_at: string
  last_updated: string
}

export default function HomePage() {
  const [chats, setChats] = useState<Chat[]>([])
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [documents, setDocuments] = useState<Document[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [isMessageLoading, setIsMessageLoading] = useState(false)
  const [metrics, setMetrics] = useState<AppMetrics | null>(null)
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false)
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)

  // Load chats and fetch metrics on mount
  useEffect(() => {
    const loadedChats = loadChatsFromStorage()
    setChats(loadedChats)
    setIsLoading(false)
    
    // Fetch app metrics
    fetchMetrics()
  }, [])

  // Save chats when they change
  useEffect(() => {
    if (!isLoading) {
      saveChatsToStorage(chats)
    }
  }, [chats, isLoading])

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch("/api/metrics")
      if (response.ok) {
        const data = await response.json()
        setMetrics(data)
      }
    } catch (error) {
      console.error("Failed to fetch metrics:", error)
    }
  }, [])

  // Load chat history from backend
  const loadChatHistoryFromBackend = useCallback(async (sessionId: string, chatId: string) => {
    if (isLoadingHistory) return
    
    setIsLoadingHistory(true)
    try {
      const response = await fetch(`/api/chat/${sessionId}`)
      if (response.ok) {
        const historyData: ChatHistoryResponse = await response.json()
        
        // Convert backend messages to frontend format
        const convertedMessages: Message[] = historyData.messages.map((msg, index) => ({
          id: `${sessionId}_${index}`,
          content: msg.content,
          role: msg.role as "user" | "assistant",
          timestamp: new Date(msg.timestamp)
        }))

        // Update the chat with loaded messages
        setChats(prev => prev.map(chat => 
          chat.id === chatId 
            ? {
                ...chat,
                messages: convertedMessages,
                session_id: sessionId
              }
            : chat
        ))

        console.log(`Loaded ${convertedMessages.length} messages from backend for session ${sessionId}`)
      }
    } catch (error) {
      console.error("Failed to load chat history from backend:", error)
    } finally {
      setIsLoadingHistory(false)
    }
  }, [isLoadingHistory])

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

  const sendMessage = async (content: string, attachments?: File[]) => {
    if (!currentChatId) {
      const newChatId = Date.now().toString()
      const newChat: Chat = {
        id: newChatId,
        title: content.slice(0, 30) + (content.length > 30 ? "..." : ""),
        messages: [],
        createdAt: new Date(),
      }
      setChats((prev) => [newChat, ...prev])
      setCurrentChatId(newChatId)
      
      // Wait for state to update then send message
      setTimeout(() => sendMessageToChat(content, newChatId, attachments), 100)
      return
    }

    sendMessageToChat(content, currentChatId, attachments)
  }

  const sendMessageToChat = async (content: string, chatId: string, attachments?: File[]) => {
    // Handle attachments if provided
    if (attachments && attachments.length > 0) {
      try {
        // Upload attachments first
        const uploadPromises = attachments.map(async (file) => {
          const formData = new FormData()
          formData.append("file", file)
          
          const response = await fetch("/api/upload", {
            method: "POST",
            body: formData,
          })
          
          const result = await response.json()
          if (!response.ok) {
            throw new Error(result.message || "Failed to upload attachment")
          }
          
          return {
            filename: file.name,
            document_id: result.document_id,
            status: result.status
          }
        })
        
        const uploadResults = await Promise.all(uploadPromises)
        console.log("Uploaded attachments:", uploadResults)
        
        // Add note about uploaded attachments to the message
        const attachmentNote = `\n\n[Uploaded attachments: ${uploadResults.map(r => r.filename).join(", ")}]`
        content += attachmentNote
        
      } catch (error) {
        console.error("Failed to upload attachments:", error)
        // Continue with the message even if attachments fail
      }
    }

    const userMessage: Message = {
      id: `${Date.now()}_user`,
      content,
      role: "user",
      timestamp: new Date(),
    }

    // Add user message to chat immediately for better UX
    setChats((prev) =>
      prev.map((chat) =>
        chat.id === chatId
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
      const currentChat = chats.find((chat) => chat.id === chatId)
      
      // Use existing session_id if available, otherwise use chatId as session
      const sessionId = currentChat?.session_id || chatId

      // Prepare chat history for context (last 10 messages)
      const chatHistory = currentChat?.messages.slice(-10).map(msg => ({
        id: msg.id,
        content: msg.content,
        role: msg.role,
        timestamp: msg.timestamp.toISOString()
      })) || []

      const requestBody = {
        message: content,
        session_id: sessionId,
        include_sources: true,
        // Include chat history for better context
        chat_history: chatHistory.length > 0 ? chatHistory : undefined
      }

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: "Network error" }))
        throw new Error(errorData.error || `HTTP ${response.status}`)
      }

      const data: ChatResponse = await response.json()

      if (data.session_id && data.response) {
        const assistantMessage: Message = {
          id: `${Date.now()}_assistant`,
          content: data.response,
          role: "assistant",
          timestamp: new Date(data.timestamp),
          sources: data.sources,
          processing_time: data.processing_time
        }

        // Update chat with session_id and assistant message
        setChats((prev) =>
          prev.map((chat) =>
            chat.id === chatId
              ? {
                  ...chat,
                  session_id: data.session_id,
                  messages: [...chat.messages, assistantMessage],
                }
              : chat,
          ),
        )

        // Update metrics
        fetchMetrics()
      } else {
        throw new Error("Invalid response format from server")
      }
    } catch (error) {
      console.error("Failed to send message:", error)
      
      const errorMessage: Message = {
        id: `${Date.now()}_error`,
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
        role: "assistant",
        timestamp: new Date(),
      }

      setChats((prev) =>
        prev.map((chat) =>
          chat.id === chatId
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
    
    // If chat has session_id but no messages loaded, try to load from backend
    const selectedChat = chats.find(chat => chat.id === chatId)
    if (selectedChat?.session_id && selectedChat.messages.length === 0) {
      loadChatHistoryFromBackend(selectedChat.session_id, chatId)
    }
  }

  const deleteChat = (chatId: string) => {
    setChats((prev) => prev.filter((chat) => chat.id !== chatId))
    if (currentChatId === chatId) {
      setCurrentChatId(null)
    }
  }

  const handleDocumentUpload = async (document: Document) => {
    try {
      // Set status to processing
      const processingDoc = { ...document, status: "processing" as const }
      setDocuments((prev) => [processingDoc, ...prev])

      // Create FormData for file upload
      const formData = new FormData()
      
      // Use the actual File object if available, otherwise create one from content
      if (document.file) {
        // This is the preferred way - using the actual File object
        formData.append("file", document.file)
      } else {
        // Fallback: create a file from document content (less reliable)
        const blob = new Blob([document.content], { type: document.type || "application/octet-stream" })
        const fileToUpload = new File([blob], document.name, { type: document.type })
        formData.append("file", fileToUpload)
      }

      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      })

      const result: DocumentUploadResponse = await response.json()

      if (response.ok) {
        // Update document status to completed
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.id === document.id
              ? { 
                  ...doc, 
                  status: "completed" as const,
                  // Store the document_id from backend for future reference
                  id: result.document_id || doc.id
                }
              : doc
          )
        )
        
        // Update metrics
        fetchMetrics()
        
        console.log(`Document uploaded successfully: ${result.filename} (${result.chunks_created} chunks)`)
      } else {
        throw new Error(result.message || "Upload failed")
      }
    } catch (error) {
      console.error("Document upload failed:", error)
      
      // Update document status to error
      setDocuments((prev) =>
        prev.map((doc) =>
          doc.id === document.id
            ? { ...doc, status: "error" as const }
            : doc
        )
      )
      
      // Show error to user (you might want to add a toast notification here)
      alert(`Failed to upload ${document.name}: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  const handleDocumentDelete = (documentId: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.id !== documentId))
    // Note: You might want to call an API endpoint to delete from backend as well
  }

  const reprocessDocuments = async () => {
    try {
      const response = await fetch("/api/documents/reprocess", {
        method: "POST",
      })

      if (response.ok) {
        const result = await response.json()
        console.log("Document reprocessing started:", result.message)
        
        // Reset document statuses
        setDocuments((prev) => 
          prev.map(doc => ({ ...doc, status: "processing" as const }))
        )
        
        // Refresh metrics after a delay
        setTimeout(() => {
          fetchMetrics()
          // Update document statuses back to completed
          setDocuments((prev) => 
            prev.map(doc => ({ ...doc, status: "completed" as const }))
          )
        }, 5000)
      } else {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to start reprocessing")
      }
    } catch (error) {
      console.error("Failed to start document reprocessing:", error)
      alert(`Failed to reprocess documents: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  // Load document info on mount
  useEffect(() => {
    const fetchDocumentInfo = async () => {
      try {
        const response = await fetch("/api/documents/info")
        if (response.ok) {
          const docInfo = await response.json()
          console.log(`Backend has ${docInfo.total_documents} documents loaded`)
        }
      } catch (error) {
        console.error("Failed to fetch document info:", error)
      }
    }

    fetchDocumentInfo()
  }, [])

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
            <div>
              <h1 className="text-xl font-semibold">Indonesian Legal Assistant</h1>
              {metrics && (
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant="secondary" className="text-xs">
                    <Brain className="w-3 h-3 mr-1" />
                    {metrics.total_documents} documents
                  </Badge>
                  <Badge variant="secondary" className="text-xs">
                    <TrendingUp className="w-3 h-3 mr-1" />
                    {metrics.total_chats} chats
                  </Badge>
                  <Badge variant="secondary" className="text-xs">
                    <Activity className="w-3 h-3 mr-1" />
                    {Math.round(metrics.uptime_seconds / 3600)}h uptime
                  </Badge>
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Dialog open={uploadDialogOpen} onOpenChange={setUploadDialogOpen}>
              <DialogTrigger asChild>
                <Button size="sm" variant="outline" className="flex items-center gap-2">
                  <Upload className="h-4 w-4" />
                  Upload Docs
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-md">
                <DialogHeader>
                  <DialogTitle>Upload Documents</DialogTitle>
                  <DialogDescription>
                    Upload PDF documents to expand the knowledge base
                  </DialogDescription>
                </DialogHeader>
                <DocumentUpload
                  onDocumentUpload={handleDocumentUpload}
                  documents={documents}
                  onDocumentDelete={handleDocumentDelete}
                />
              </DialogContent>
            </Dialog>
            <Button 
              onClick={reprocessDocuments} 
              size="sm" 
              variant="outline"
              className="flex items-center gap-2"
            >
              <Activity className="h-4 w-4" />
              Reprocess
            </Button>
            <Button onClick={createNewChat} size="sm" className="flex items-center gap-2">
              <Plus className="h-4 w-4" />
              New Chat
            </Button>
          </div>
        </header>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4">
          {isLoadingHistory && (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary mr-2"></div>
              <span className="text-sm text-muted-foreground">Loading chat history...</span>
            </div>
          )}
          
          {currentChat?.messages.length === 0 || !currentChat ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-2xl">
                <div className="mb-8">
                  <Brain className="w-16 h-16 mx-auto text-primary/60 mb-4" />
                </div>
                <h2 className="text-3xl font-bold mb-4">Indonesian Legal Assistant</h2>
                <p className="text-muted-foreground mb-8 text-lg">
                  Ask me about Indonesian laws and regulations. I can help you understand 
                  legal documents, procedures, and provide guidance based on official sources.
                </p>
                
                {/* Stats Cards */}
                {metrics && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Documents</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold">{metrics.total_documents}</div>
                        <p className="text-xs text-muted-foreground">
                          {metrics.documents_processed} processed
                        </p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Chats</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold">{metrics.total_chats}</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Success Rate</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold">
                          {metrics.successful_responses + metrics.failed_responses > 0
                            ? Math.round((metrics.successful_responses / (metrics.successful_responses + metrics.failed_responses)) * 100)
                            : 0}%
                        </div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Uptime</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold">
                          {Math.round(metrics.uptime_seconds / 3600)}h
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}

                {/* Example Topics */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
                  <Card className="p-4 hover:shadow-md transition-shadow cursor-pointer"
                        onClick={() => sendMessage("Apa itu hukum perdata Indonesia?")}>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                        <FileText className="w-4 h-4 text-blue-600" />
                      </div>
                      <h3 className="font-medium">Civil Law</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Questions about Indonesian civil law, contracts, and property rights
                    </p>
                  </Card>

                  <Card className="p-4 hover:shadow-md transition-shadow cursor-pointer"
                        onClick={() => sendMessage("Bagaimana proses hukum pidana di Indonesia?")}>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-8 h-8 rounded-full bg-red-100 flex items-center justify-center">
                        <Activity className="w-4 h-4 text-red-600" />
                      </div>
                      <h3 className="font-medium">Criminal Law</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Criminal procedures, rights, and legal processes in Indonesia
                    </p>
                  </Card>

                  <Card className="p-4 hover:shadow-md transition-shadow cursor-pointer"
                        onClick={() => sendMessage("Apa saja peraturan bisnis di Indonesia?")}>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
                        <TrendingUp className="w-4 h-4 text-green-600" />
                      </div>
                      <h3 className="font-medium">Business Law</h3>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Corporate regulations, business licensing, and commercial law
                    </p>
                  </Card>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto space-y-4">
              {currentChat.messages.map((message) => (
                <div key={message.id} className="space-y-2">
                  <ChatMessage message={message} />
                  {message.role === "assistant" && (message.sources || message.processing_time) && (
                    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground ml-12">
                      {message.processing_time && (
                        <Badge variant="outline" className="text-xs">
                          <Clock className="w-3 h-3 mr-1" />
                          {message.processing_time.toFixed(2)}s
                        </Badge>
                      )}
                      {message.sources && message.sources.length > 0 && (
                        <Badge variant="outline" className="text-xs">
                          <FileText className="w-3 h-3 mr-1" />
                          {message.sources.length} sources
                        </Badge>
                      )}
                      {message.sources && message.sources.length > 0 && (
                        <details className="text-xs">
                          <summary className="cursor-pointer text-primary hover:underline">
                            View sources
                          </summary>
                          <div className="mt-2 space-y-1 max-w-md">
                            {message.sources.slice(0, 3).map((source, idx) => (
                              <div key={idx} className="p-2 bg-muted rounded text-xs">
                                <div className="font-medium">
                                  {source.metadata?.title || source.source}
                                </div>
                                <div className="text-muted-foreground line-clamp-2">
                                  {source.content.slice(0, 100)}...
                                </div>
                                <div className="text-primary">Score: {(source.score * 100).toFixed(1)}%</div>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Chat Input */}
        <div className="border-t border-border p-4">
          <div className="w-full mx-auto">
            <ChatInput 
              onSendMessage={sendMessage} 
              disabled={isMessageLoading}
              placeholder={
                currentChat?.messages.length 
                  ? "Continue your legal consultation..."
                  : "Ask a question about Indonesian law..."
              }
              showAttachButton={true}
              maxAttachments={2}
            />
            {isMessageLoading && (
              <div className="flex items-center justify-center mt-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary mr-2"></div>
                <span className="text-sm text-muted-foreground">
                  Searching legal documents and generating response...
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}