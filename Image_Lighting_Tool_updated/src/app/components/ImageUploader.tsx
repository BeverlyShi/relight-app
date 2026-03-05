import { Upload, X } from "lucide-react";
import { useCallback, useState } from "react";

interface ImageUploaderProps {
  onImageUpload: (file: File) => void;
  uploadedImage: string | null;
  onClear: () => void;
}

export function ImageUploader({ onImageUpload, uploadedImage, onClear }: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      onImageUpload(files[0]);
    }
  }, [onImageUpload]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onImageUpload(files[0]);
    }
  }, [onImageUpload]);

  if (uploadedImage) {
    return (
      <div className="relative rounded-xl overflow-hidden bg-white shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
        <img src={uploadedImage} alt="Uploaded" className="w-full h-48 object-cover" />
        <button
          onClick={onClear}
          className="absolute top-3 right-3 bg-black/60 backdrop-blur-md hover:bg-black/80 text-white rounded-full p-2 transition-all shadow-lg hover:scale-110"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`rounded-xl p-10 text-center transition-all duration-300 cursor-pointer ${
        isDragging 
          ? 'bg-gradient-to-br from-blue-50 to-indigo-50 shadow-2xl -translate-y-1 scale-[1.02]' 
          : 'bg-white shadow-lg hover:shadow-2xl hover:-translate-y-2'
      }`}
    >
      <input
        type="file"
        accept="image/*"
        onChange={handleFileInput}
        className="hidden"
        id="file-upload"
      />
      <label htmlFor="file-upload" className="cursor-pointer block">
        <div className={`w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center transition-all duration-300 ${
          isDragging 
            ? 'bg-gradient-to-br from-blue-500 to-indigo-500 shadow-lg' 
            : 'bg-gradient-to-br from-blue-100 to-indigo-100'
        }`}>
          <Upload className={`w-7 h-7 transition-colors ${
            isDragging ? 'text-white' : 'text-blue-600'
          }`} />
        </div>
        <p className="text-sm text-gray-700 mb-1">拖拽图片到此处或点击上传</p>
        <p className="text-xs text-gray-500">支持 JPG、PNG 等格式</p>
      </label>
    </div>
  );
}
