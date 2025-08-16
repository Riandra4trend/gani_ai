"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Upload, File, X, FileText, FileImage, FileVideo, AlertCircle, CheckCircle } from "lucide-react"
import { cn } from "@/lib/utils"

interface Document {
  id: string
  name: string
  size: number
  type: string
  content: string
  uploadedAt: Date
  status?: "processing" | "completed" | "error"
  file?: File  // Store actual File object for upload
}

interface DocumentUploadProps {
  onDocumentUpload: (document: Document) => void
  documents: Document[]
  onDocumentDelete: (documentId: string) => void
}

export function DocumentUpload({ onDocumentUpload, documents, onDocumentDelete }: DocumentUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<{ [key: string]: number }>({})

  const getFileIcon = (type: string, status?: string) => {
    if (status === "processing") return <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary border-t-transparent" />
    if (status === "error") return <AlertCircle className="h-4 w-4 text-destructive" />
    if (status === "completed") return <CheckCircle className="h-4 w-4 text-green-600" />
    
    if (type.startsWith("image/")) return <FileImage className="h-4 w-4" />
    if (type.startsWith("video/")) return <FileVideo className="h-4 w-4" />
    return <FileText className="h-4 w-4" />
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const validateFile = (file: File): { valid: boolean; error?: string } => {
    // Check file type
    const allowedTypes = [
      'application/pdf',
      'text/plain',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/markdown',
      'text/csv',
      'application/json'
    ]

    const allowedExtensions = ['.pdf', '.txt', '.doc', '.docx', '.md', '.csv', '.json']

    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase()
    
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      return {
        valid: false,
        error: `File type not supported. Please upload PDF, TXT, DOC, DOCX, MD, CSV, or JSON files.`
      }
    }

    // Check file size (max 50MB)
    const maxSize = 50 * 1024 * 1024 // 50MB in bytes
    if (file.size > maxSize) {
      return {
        valid: false,
        error: `File is too large. Maximum size is 50MB.`
      }
    }

    return { valid: true }
  }

  const processFile = async (file: File) => {
    const fileId = `${Date.now()}_${file.name}`
    
    // Validate file
    const validation = validateFile(file)
    if (!validation.valid) {
      alert(validation.error)
      return
    }

    setUploadProgress((prev) => ({ ...prev, [fileId]: 0 }))

    try {
      let content = ""

      // Simulate upload progress for UI feedback
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          const currentProgress = prev[fileId] || 0
          if (currentProgress >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return { ...prev, [fileId]: currentProgress + 10 }
        })
      }, 100)

      // Read file content based on type
      if (file.type === "text/plain" || file.name.endsWith(".txt") || file.name.endsWith(".md")) {
        content = await file.text()
      } else if (file.type === "application/json" || file.name.endsWith(".json")) {
        content = await file.text()
      } else if (file.type === "text/csv" || file.name.endsWith(".csv")) {
        content = await file.text()
      } else if (file.type === "application/pdf") {
        content = `[PDF Document: ${file.name}] - Content will be extracted by the backend`
      } else if (
        file.type === "application/msword" || 
        file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
        file.name.endsWith(".doc") || 
        file.name.endsWith(".docx")
      ) {
        content = `[Word Document: ${file.name}] - Content will be extracted by the backend`
      } else {
        content = `[File: ${file.name}] - Binary file uploaded for processing`
      }

      // Complete progress
      clearInterval(progressInterval)
      setUploadProgress((prev) => ({ ...prev, [fileId]: 100 }))

      const document: Document = {
        id: fileId,
        name: file.name,
        size: file.size,
        type: file.type,
        content,
        uploadedAt: new Date(),
        status: "processing",
        file: file // Store the actual File object for backend upload
      }

      onDocumentUpload(document)
      
      // Remove progress after a short delay
      setTimeout(() => {
        setUploadProgress((prev) => {
          const newProgress = { ...prev }
          delete newProgress[fileId]
          return newProgress
        })
      }, 1000)

    } catch (error) {
      console.error("Error processing file:", error)
      setUploadProgress((prev) => {
        const newProgress = { ...prev }
        delete newProgress[fileId]
        return newProgress
      })
      alert(`Error processing ${file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    files.forEach(processFile)
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    files.forEach(processFile)
    // Reset the input value to allow uploading the same file again
    e.target.value = ''
  }

  const getStatusColor = (status?: string) => {
    switch (status) {
      case "processing":
        return "text-yellow-600"
      case "completed":
        return "text-green-600"
      case "error":
        return "text-destructive"
      default:
        return "text-muted-foreground"
    }
  }

  const getStatusText = (status?: string) => {
    switch (status) {
      case "processing":
        return "Processing..."
      case "completed":
        return "Completed"
      case "error":
        return "Error"
      default:
        return "Ready"
    }
  }

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      <Card
        className={cn(
          "border-2 border-dashed p-6 text-center transition-colors cursor-pointer",
          isDragOver ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50",
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => document.getElementById("file-upload")?.click()}
      >
        <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm font-medium mb-1">Drop files here or click to upload</p>
        <p className="text-xs text-muted-foreground mb-2">Supports PDF, TXT, DOCX, MD, CSV, JSON (max 50MB)</p>
        <p className="text-xs text-muted-foreground">
          PDF documents will be processed and added to the legal knowledge base
        </p>
        <input
          id="file-upload"
          type="file"
          multiple
          className="hidden"
          onChange={handleFileSelect}
          accept=".pdf,.txt,.docx,.doc,.md,.csv,.json,application/pdf,text/plain,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        />
      </Card>

      {/* Upload Progress */}
      {Object.entries(uploadProgress).map(([fileId, progress]) => (
        <Card key={fileId} className="p-3">
          <div className="flex items-center gap-2 mb-2">
            <File className="h-4 w-4" />
            <span className="text-sm font-medium">Preparing upload...</span>
          </div>
          <Progress value={progress} className="h-2" />
          <div className="text-xs text-muted-foreground mt-1">{progress}%</div>
        </Card>
      ))}

      {/* Document List */}
      {documents.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">
            Documents ({documents.length})
          </h3>
          <div className="max-h-60 overflow-y-auto space-y-2">
            {documents.map((doc) => (
              <Card key={doc.id} className="p-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    {getFileIcon(doc.type, doc.status)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{doc.name}</p>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>{formatFileSize(doc.size)}</span>
                        <span>•</span>
                        <span>{doc.uploadedAt.toLocaleDateString()}</span>
                        <span>•</span>
                        <span className={getStatusColor(doc.status)}>
                          {getStatusText(doc.status)}
                        </span>
                      </div>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => onDocumentDelete(doc.id)}
                    className="h-6 w-6 text-muted-foreground hover:text-destructive"
                    disabled={doc.status === "processing"}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
                {doc.status === "error" && (
                  <div className="mt-2 text-xs text-destructive bg-destructive/10 p-2 rounded">
                    Upload failed. Please try again or contact support.
                  </div>
                )}
              </Card>
            ))}
          </div>
        </div>
      )}
      
      {documents.length > 0 && (
        <div className="text-xs text-muted-foreground text-center pt-2 border-t">
          💡 Uploaded documents are automatically processed and added to the knowledge base
        </div>
      )}
    </div>
  )
}