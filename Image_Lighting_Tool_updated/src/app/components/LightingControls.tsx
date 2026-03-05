import { Sun, ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from "lucide-react";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";

interface LightingControlsProps {
  angle: number;
  onAngleChange: (angle: number) => void;
  brightness: number;
  onBrightnessChange: (value: number) => void;
  temperature: number;
  onTemperatureChange: (value: number) => void;
  intensity: number;
  onIntensityChange: (value: number) => void;
  prompt: string;
  onPromptChange: (value: string) => void;
  onGenerate: () => void;
  isGenerating: boolean;
  hasImage: boolean;
}

export function LightingControls({
  angle,
  onAngleChange,
  brightness,
  onBrightnessChange,
  temperature,
  onTemperatureChange,
  intensity,
  onIntensityChange,
  prompt,
  onPromptChange,
  onGenerate,
  isGenerating,
  hasImage
}: LightingControlsProps) {
  
  const presets = [
    { name: '顶光', angle: 0, icon: ArrowUp },
    { name: '右光', angle: 90, icon: ArrowRight },
    { name: '底光', angle: 180, icon: ArrowDown },
    { name: '左光', angle: 270, icon: ArrowLeft },
  ];

  const handleDirectionClick = (x: number, y: number) => {
    const centerX = 80;
    const centerY = 80;
    const dx = x - centerX;
    const dy = y - centerY;
    let calculatedAngle = Math.atan2(dy, dx) * (180 / Math.PI) + 90;
    if (calculatedAngle < 0) calculatedAngle += 360;
    onAngleChange(Math.round(calculatedAngle));
  };

  const getIndicatorPosition = () => {
    const radius = 70;
    const angleRad = ((angle - 90) * Math.PI) / 180;
    const x = 80 + radius * Math.cos(angleRad);
    const y = 80 + radius * Math.sin(angleRad);
    return { x, y };
  };

  const indicatorPos = getIndicatorPosition();

  return (
    <div className="space-y-6">
      {/* 光照方向设置 */}
      <div>
        <Label className="mb-3 block">光照方向</Label>
        <div className="flex justify-center mb-4">
          <svg
            width="160"
            height="160"
            className="cursor-pointer drop-shadow-lg"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const x = e.clientX - rect.left;
              const y = e.clientY - rect.top;
              handleDirectionClick(x, y);
            }}
          >
            {/* 外圈 */}
            <circle cx="80" cy="80" r="75" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="2" />
            {/* 内圈 - 毛玻璃效果 */}
            <circle cx="80" cy="80" r="60" fill="rgba(255,255,255,0.4)" />
            
            {/* 刻度线 */}
            {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => {
              const angleRad = ((deg - 90) * Math.PI) / 180;
              const x1 = 80 + 60 * Math.cos(angleRad);
              const y1 = 80 + 60 * Math.sin(angleRad);
              const x2 = 80 + 70 * Math.cos(angleRad);
              const y2 = 80 + 70 * Math.sin(angleRad);
              return (
                <line
                  key={deg}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke="#d1d5db"
                  strokeWidth="1.5"
                />
              );
            })}

            {/* 中心点 */}
            <circle cx="80" cy="80" r="4" fill="#6b7280" />
            
            {/* 光照指示器 */}
            <circle cx={indicatorPos.x} cy={indicatorPos.y} r="8" fill="#3b82f6" className="drop-shadow-lg" />
            <line x1="80" y1="80" x2={indicatorPos.x} y2={indicatorPos.y} stroke="#3b82f6" strokeWidth="2" opacity="0.8" />
          </svg>
        </div>
        <div className="text-center text-sm text-gray-600 mb-4">{angle}°</div>
        
        {/* 预设快捷键 */}
        <div className="grid grid-cols-2 gap-2">
          {presets.map((preset) => {
            const Icon = preset.icon;
            return (
              <Button
                key={preset.name}
                variant={angle === preset.angle ? "default" : "outline"}
                onClick={() => onAngleChange(preset.angle)}
                className="w-full"
              >
                <Icon className="w-4 h-4 mr-2" />
                {preset.name}
              </Button>
            );
          })}
        </div>
      </div>

      {/* 光照参数 */}
      <div className="space-y-4">
        <div>
          <div className="flex justify-between items-center mb-2">
            <Label>亮度</Label>
            <span className="text-sm text-gray-600">{brightness}</span>
          </div>
          <Slider
            value={[brightness]}
            onValueChange={(value) => onBrightnessChange(value[0])}
            min={0}
            max={100}
            step={1}
            className="w-full"
          />
        </div>

        <div>
          <div className="flex justify-between items-center mb-2">
            <Label>色温</Label>
            <span className="text-sm text-gray-600">{temperature}K</span>
          </div>
          <Slider
            value={[temperature]}
            onValueChange={(value) => onTemperatureChange(value[0])}
            min={2000}
            max={8000}
            step={100}
            className="w-full"
          />
        </div>

        <div>
          <div className="flex justify-between items-center mb-2">
            <Label>强度</Label>
            <span className="text-sm text-gray-600">{intensity}</span>
          </div>
          <Slider
            value={[intensity]}
            onValueChange={(value) => onIntensityChange(value[0])}
            min={0}
            max={100}
            step={1}
            className="w-full"
          />
        </div>
      </div>

      {/* Prompt 输入 */}
      <div>
        <Label htmlFor="prompt" className="mb-2 block">提示词（Prompt）</Label>
        <Textarea
          id="prompt"
          placeholder="描述您想要的光照效果..."
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          rows={4}
          className="resize-none"
        />
      </div>

      {/* 生成按钮 */}
      <Button
        onClick={onGenerate}
        disabled={!hasImage || isGenerating}
        className="w-full"
        size="lg"
      >
        <Sun className="w-5 h-5 mr-2" />
        {isGenerating ? '生成中...' : '生成打光效果'}
      </Button>
    </div>
  );
}
