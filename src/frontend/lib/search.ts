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

export interface SearchResult {
  chat: Chat
  matchType: "title" | "message"
  matchedMessage?: Message
  relevanceScore: number
}

export function searchChats(chats: Chat[], query: string): SearchResult[] {
  if (!query.trim()) return []

  const searchTerm = query.toLowerCase().trim()
  const results: SearchResult[] = []

  chats.forEach((chat) => {
    let relevanceScore = 0
    let matchType: "title" | "message" = "title"
    let matchedMessage: Message | undefined

    // Check title match
    if (chat.title.toLowerCase().includes(searchTerm)) {
      relevanceScore += 10
      if (chat.title.toLowerCase().startsWith(searchTerm)) {
        relevanceScore += 5
      }
    }

    // Check message content matches
    let messageMatches = 0
    let bestMessageMatch: Message | undefined

    chat.messages.forEach((message) => {
      if (message.content.toLowerCase().includes(searchTerm)) {
        messageMatches++
        if (!bestMessageMatch || message.role === "user") {
          bestMessageMatch = message
        }

        // Higher score for exact word matches
        const words = searchTerm.split(" ")
        words.forEach((word) => {
          if (message.content.toLowerCase().includes(word)) {
            relevanceScore += 2
          }
        })
      }
    })

    if (messageMatches > 0) {
      matchType = "message"
      matchedMessage = bestMessageMatch
      relevanceScore += messageMatches * 3
    }

    // Add recency bonus
    const daysSinceCreated = (Date.now() - chat.createdAt.getTime()) / (1000 * 60 * 60 * 24)
    relevanceScore += Math.max(0, 5 - daysSinceCreated)

    if (relevanceScore > 0) {
      results.push({
        chat,
        matchType,
        matchedMessage,
        relevanceScore,
      })
    }
  })

  return results.sort((a, b) => b.relevanceScore - a.relevanceScore)
}

export function getSearchSuggestions(chats: Chat[], currentQuery: string): string[] {
  const suggestions = new Set<string>()
  const query = currentQuery.toLowerCase()

  chats.forEach((chat) => {
    // Add title words as suggestions
    const titleWords = chat.title.toLowerCase().split(/\s+/)
    titleWords.forEach((word) => {
      if (word.length > 2 && word.startsWith(query) && word !== query) {
        suggestions.add(word)
      }
    })

    // Add common message words as suggestions
    chat.messages.forEach((message) => {
      const words = message.content.toLowerCase().split(/\s+/)
      words.forEach((word) => {
        if (word.length > 3 && word.startsWith(query) && word !== query) {
          suggestions.add(word)
        }
      })
    })
  })

  return Array.from(suggestions).slice(0, 5)
}
