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

const STORAGE_KEY = "ai-assistant-chats"

export const saveChatsToStorage = (chats: Chat[]) => {
  try {
    const serializedChats = JSON.stringify(chats, (key, value) => {
      if (key === "timestamp" || key === "createdAt") {
        return value instanceof Date ? value.toISOString() : value
      }
      return value
    })
    localStorage.setItem(STORAGE_KEY, serializedChats)
  } catch (error) {
    console.error("Failed to save chats to localStorage:", error)
  }
}

export const loadChatsFromStorage = (): Chat[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (!stored) return []

    const parsed = JSON.parse(stored)
    return parsed.map((chat: any) => ({
      ...chat,
      createdAt: new Date(chat.createdAt),
      messages: chat.messages.map((message: any) => ({
        ...message,
        timestamp: new Date(message.timestamp),
      })),
    }))
  } catch (error) {
    console.error("Failed to load chats from localStorage:", error)
    return []
  }
}

export const clearChatsFromStorage = () => {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch (error) {
    console.error("Failed to clear chats from localStorage:", error)
  }
}
