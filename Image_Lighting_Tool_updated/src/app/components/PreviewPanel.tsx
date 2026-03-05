import { RefreshCw } from "lucide-react";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";

interface PreviewPanelProps {
  originalImage: string | null;
  generatedImage: string | null;
  isGenerating: boolean;
  progress: number;
  onRegenerate: () => void;
}

export function PreviewPanel({
  originalImage,
  generatedImage,
  isGenerating,
  progress,
  onRegenerate
}: PreviewPanelProps) {
  
  if (!originalImage && !generatedImage) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center bg-white rounded-xl p-8 border border-gray-200 shadow-lg">
          <p className="mb-2 text-gray-600">暂无预览</p>
          <p className="text-sm text-gray-500">请先上传图片</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* 图片对比区域 */}
      <div className="flex-1 grid grid-cols-2 gap-4 mb-4">
        {/* 原图 */}
        <div className="flex flex-col">
          <div className="text-sm mb-2 text-gray-700">原图</div>
          <div className="flex-1 bg-white rounded-xl border border-gray-200 overflow-hidden flex items-center justify-center shadow-lg">
            {originalImage ? (
              <img
                src={originalImage}
                alt="Original"
                className="max-w-full max-h-full object-contain"
              />
            ) : (
              <div className="text-gray-400 text-sm">无图片</div>
            )}
          </div>
        </div>

        {/* 生成结果 */}
        <div className="flex flex-col">
          <div className="text-sm mb-2 text-gray-700">打光效果</div>
          <div className="flex-1 bg-white rounded-xl border border-gray-200 overflow-hidden flex items-center justify-center relative shadow-lg">
            {isGenerating ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 backdrop-blur-xl">
                <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4 drop-shadow-lg"></div>
                <p className="text-sm text-gray-700 mb-2">生成中...</p>
                <div className="w-48">
                  <Progress value={progress} className="h-2" />
                </div>
                <p className="text-xs text-gray-600 mt-2">{progress}%</p>
              </div>
            ) : generatedImage ? (
              <img
                src={generatedImage}
                alt="Generated"
                className="max-w-full max-h-full object-contain"
              />
            ) : (
              <div className="text-gray-500 text-sm">等待生成</div>
            )}
          </div>
        </div>
      </div>

      {/* 重新生成按钮 */}
      {generatedImage && !isGenerating && (
        <div className="flex justify-center">
          <Button
            onClick={onRegenerate}
            variant="outline"
            size="lg"
            className="bg-white hover:bg-gray-50 border-gray-200 shadow-lg"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            重新生成
          </Button>
        </div>
      )}
    </div>
  );
}
