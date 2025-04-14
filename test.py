import cv2
import numpy as np

# 生成全黑图像
test_img = np.zeros((100, 100), dtype=np.uint8)

# 保存并验证
cv2.imwrite("test.png", test_img)
print("保存成功！")