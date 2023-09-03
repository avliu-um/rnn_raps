import time
from datetime import datetime


i = 0

print(f'start!')

while True:
    print(datetime.now())
    time.sleep(10)
    i += 1
    if i == 120:
        break

print(f'done!')
