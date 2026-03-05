import { useState, useCallback } from "react";
import { Lightbulb } from "lucide-react";
import { ImageUploader } from "./components/ImageUploader";
import { LightingControls } from "./components/LightingControls";
import { PreviewPanel } from "./components/PreviewPanel";

export default function App() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // 光照参数
  const [angle, setAngle] = useState(0);
  const [brightness, setBrightness] = useState(50);
  const [temperature, setTemperature] = useState(5000);
  const [intensity, setIntensity] = useState(50);
  const [prompt, setPrompt] = useState("");

  const handleImageUpload = useCallback((file: File) => {
    setUploadedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string);
      setGeneratedImage(null);
      setErrorMsg(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleClearImage = useCallback(() => {
    setUploadedFile(null);
    setUploadedImage(null);
    setGeneratedImage(null);
    setErrorMsg(null);
  }, []);

  const callRelightAPI = useCallback(async () => {
    if (!uploadedFile) return;

    setIsGenerating(true);
    setProgress(0);
    setGeneratedImage(null);
    setErrorMsg(null);

    const interval = setInterval(() => {
      setProgress((prev) => (prev < 90 ? prev + 3 : prev));
    }, 1000);

    try {
      const formData = new FormData();
      formData.append("file", uploadedFile);
      formData.append("angle", String(angle));
      formData.append("prompt", prompt || "natural lighting");
      formData.append("steps", "25");
      formData.append("cfg", String((intensity / 100) * 4));
      formData.append("seed", "12345");
      formData.append("highres_denoise", "0.3");

      const response = await fetch("http://localhost:6007/relight", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API 请求失败: ${response.status}`);
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);

      clearInterval(interval);
      setProgress(100);
      setGeneratedImage(imageUrl);
    } catch (err) {
      clearInterval(interval);
      setErrorMsg(err instanceof Error ? err.message : "生成失败，请重试");
    } finally {
      setTimeout(() => setIsGenerating(false), 300);
    }
  }, [uploadedFile, angle, prompt, intensity]);

  const handleGenerate = useCallback(() => {
    if (!uploadedFile) return;
    callRelightAPI();
  }, [uploadedFile, callRelightAPI]);

  const handleRegenerate = useCallback(() => {
    callRelightAPI();
  }, [callRelightAPI]);

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 relative overflow-hidden p-4">
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-400/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-400/20 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-indigo-400/20 rounded-full blur-3xl"></div>
      </div>

      <header className="bg-white/70 backdrop-blur-xl rounded-xl border border-white/30 px-6 py-4 mb-4 relative z-10 shadow-lg">
        <div className="flex items-center">
          <Lightbulb className="w-6 h-6 text-blue-600 mr-3" />
          <h1 className="text-xl">智能打光系统</h1>
        </div>
      </header>

      <div className="flex-1 flex gap-4 overflow-hidden relative z-10">
        <aside className="w-96 bg-white/70 backdrop-blur-xl rounded-xl border border-white/30 p-6 overflow-y-auto shadow-lg">
          <div className="space-y-6">
            <div>
              <h2 className="mb-4">上传图片</h2>
              <ImageUploader
                onImageUpload={handleImageUpload}
                uploadedImage={uploadedImage}
                onClear={handleClearImage}
              />
            </div>

            {uploadedImage && (
              <div>
                <h2 className="mb-4">光照设置</h2>
                <LightingControls
                  angle={angle}
                  onAngleChange={setAngle}
                  brightness={brightness}
                  onBrightnessChange={setBrightness}
                  temperature={temperature}
                  onTemperatureChange={setTemperature}
                  intensity={intensity}
                  onIntensityChange={setIntensity}
                  prompt={prompt}
                  onPromptChange={setPrompt}
                  onGenerate={handleGenerate}
                  isGenerating={isGenerating}
                  hasImage={!!uploadedImage}
                />
              </div>
            )}

            {errorMsg && (
              <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl px-4 py-3 text-sm">
                {errorMsg}
              </div>
            )}
          </div>
        </aside>

        <main className="flex-1 overflow-hidden">
          <div className="h-full bg-white/70 backdrop-blur-xl rounded-xl border border-white/30 p-6 shadow-lg">
            <h2 className="mb-4">效果预览</h2>
            <div className="h-[calc(100%-3rem)]">
              <PreviewPanel
                originalImage={uploadedImage}
                generatedImage={generatedImage}
                isGenerating={isGenerating}
                progress={progress}
                onRegenerate={handleRegenerate}
              />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
