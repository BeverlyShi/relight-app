import requests
from PIL import Image
import io

with open("/Users/shiboyang/Downloads/u7344393522_httpss.mj.runxNPgAvSFhz4_Sci-fi_Cyberpunk_high-fi_ff9dd63b-03bc-49b3-98f5-038125ba7bd9_1.png", "rb") as f:
    response = requests.post(
        "http://localhost:6007/relight",
        files={"file": ("test.jpg", f, "image/jpeg")},
        data={
            "angle": 90,
            "prompt": "magic lit",
            "steps": 25,
        }
    )

print("状态码:", response.status_code)
result = Image.open(io.BytesIO(response.content))
result.save("result.png")
result.show()
print("✅ 结果已保存到 result.png")
