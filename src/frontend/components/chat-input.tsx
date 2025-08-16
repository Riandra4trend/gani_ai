"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Send, Paperclip, X, FileText } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface ChatInputProps {
  onSendMessage: (message: string, attachments?: File[]) => void
  disabled?: boolean
  placeholder?: string
  showAttachButton?: boolean
  maxAttachments?: number
}

interface AttachedFile {
  file: File
  id: string
}

export function ChatInput({ 
  onSendMessage, 
  disabled = false, 
  placeholder = "Type your message here...",
  showAttachButton = false,
  maxAttachments = 3
}: ChatInputProps) {
  const [message, setMessage] = useState("")
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if ((message.trim() || attachedFiles.length > 0) && !disabled) {
      const files = attachedFiles.map(af => af.file)
      onSendMessage(message.trim(), files.length > 0 ? files : undefined)
      setMessage("")
      setAttachedFiles([])
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleFileAttach = () => {
    fileInputRef.current?.click()
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    
    // Check if adding files would exceed limit
    if (attachedFiles.length + files.length > maxAttachments) {
      alert(`You can only attach up to ${maxAttachments} files at once.`)
      return
    }

    const newAttachedFiles = files.map(file => ({
      file,
      id: `${Date.now()}_${file.name}`
    }))

    setAttachedFiles(prev => [...prev, ...newAttachedFiles])
    
    // Reset input
    e.target.value = ''
  }

  const removeAttachment = (id: string) => {
    setAttachedFiles(prev => prev.filter(af => af.id !== id))
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const canSubmit = (message.trim().length > 0 || attachedFiles.length > 0) && !disabled

  return (
    <div className="mx-8 space-y-2">
      {/* File Attachments */}
      {attachedFiles.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {attachedFiles.map((attachedFile) => (
            <Badge key={attachedFile.id} variant="secondary" className="flex items-center gap-1 px-2 py-1">
              <FileText className="h-3 w-3" />
              <span className="text-xs max-w-32 truncate">
                {attachedFile.file.name}
              </span>
              <span className="text-xs opacity-70">
                ({formatFileSize(attachedFile.file.size)})
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-4 w-4 p-0 hover:bg-destructive/20"
                onClick={() => removeAttachment(attachedFile.id)}
                type="button"
              >
                <X className="h-2 w-2" />
              </Button>
            </Badge>
          ))}
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="flex-1 relative">
          <Textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="min-h-[50px] max-h-[200px] w-full resize-none pr-20"
            rows={1}
            disabled={disabled}
          />
          <div className="absolute right-2 bottom-2 flex gap-1">
            {showAttachButton && (
              <Button
                type="button"
                size="icon"
                variant="ghost"
                onClick={handleFileAttach}
                disabled={disabled || attachedFiles.length >= maxAttachments}
                className="h-8 w-8"
                title={
                  attachedFiles.length >= maxAttachments 
                    ? `Maximum ${maxAttachments} files allowed`
                    : "Attach files"
                }
              >
                <Paperclip className="h-4 w-4" />
              </Button>
            )}
            <Button
              type="submit"
              size="icon"
              disabled={!canSubmit}
              className="h-8 w-8"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Hidden file input */}
        {showAttachButton && (
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={handleFileSelect}
            accept=".pdf,.txt,.docx,.doc,.md,.csv,.json,application/pdf,text/plain,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          />
        )}
      </form>
    </div>
  )
}