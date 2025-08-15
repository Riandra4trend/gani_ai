"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Upload, File, X, FileText, FileImage, FileVideo } from "lucide-react"
import { cn } from "@/lib/utils"

interface Document {
  id: string
  name: string
  size: number
  type: string
  content: string
  uploadedAt: Date
}

interface DocumentUploadProps {
  onDocumentUpload: (document: Document) => void
  documents: Document[]
  onDocumentDelete: (documentId: string) => void
}

export function DocumentUpload({ onDocumentUpload, documents, onDocumentDelete }: DocumentUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<{ [key: string]: number }>({})

  const getFileIcon = (type: string) => {
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

  const processFile = async (file: File) => {
    const fileId = Date.now().toString()
    setUploadProgress((prev) => ({ ...prev, [fileId]: 0 }))

    try {
      let content = ""

      // Simulate upload progress
      for (let i = 0; i <= 100; i += 10) {
        setUploadProgress((prev) => ({ ...prev, [fileId]: i }))
        await new Promise((resolve) => setTimeout(resolve, 50))
      }

      // Read file content based on type
      if (file.type.startsWith("text/") || file.name.endsWith(".txt")) {
        content = await file.text()
      } else if (file.type === "application/pdf") {
        content = `[PDF Document: ${file.name}] - Content will be extracted for vector processing`
      } else if (file.type.includes("document") || file.name.endsWith(".docx")) {
        content = `[Document: ${file.name}] - Content will be extracted for vector processing`
      } else {
        content = `[File: ${file.name}] - Binary file uploaded for processing`
      }

      const document: Document = {
        id: fileId,
        name: file.name,
        size: file.size,
        type: file.type,
        content,
        uploadedAt: new Date(),
      }

      onDocumentUpload(document)
      setUploadProgress((prev) => {
        const newProgress = { ...prev }
        delete newProgress[fileId]
        return newProgress
      })
    } catch (error) {
      console.error("Error processing file:", error)
      setUploadProgress((prev) => {
        const newProgress = { ...prev }
        delete newProgress[fileId]
        return newProgress
      })
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
        <p className="text-xs text-muted-foreground">Supports PDF, TXT, DOCX, and more</p>
        <input
          id="file-upload"
          type="file"
          multiple
          className="hidden"
          onChange={handleFileSelect}
          accept=".pdf,.txt,.docx,.doc,.md,.csv,.json"
        />
      </Card>

      {/* Upload Progress */}
      {Object.entries(uploadProgress).map(([fileId, progress]) => (
        <Card key={fileId} className="p-3">
          <div className="flex items-center gap-2 mb-2">
            <File className="h-4 w-4" />
            <span className="text-sm">Uploading...</span>
          </div>
          <Progress value={progress} className="h-2" />
        </Card>
      ))}

      {/* Document List */}
      {documents.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Uploaded Documents</h3>
          {documents.map((doc) => (
            <Card key={doc.id} className="p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  {getFileIcon(doc.type)}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{doc.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(doc.size)} • {doc.uploadedAt.toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => onDocumentDelete(doc.id)}
                  className="h-6 w-6 text-muted-foreground hover:text-destructive"
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
