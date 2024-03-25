import pyautogui
import time

countdown_seconds = 4000

try:
    time.sleep(countdown_seconds)

    pyautogui.moveTo(1466, 633, duration=0.5)

    pyautogui.click()

except KeyboardInterrupt:
    # 如果在倒计时期间按下Ctrl+C，则取消操作
    print("操作已取消")
