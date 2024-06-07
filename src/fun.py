import time

def main():
    while True:
        try:
            # 模拟长时间运行的任务
            print("运行中...按 Ctrl+C 尝试停止")
            time.sleep(2)
        except KeyboardInterrupt:
            print("哈哈，关不掉吧，急死你")

if __name__ == "__main__":
    main()
