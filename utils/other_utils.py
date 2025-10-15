import time
import shortuuid

def get_unique_id():
    return f"{int(time.time() * 1e7)}-{shortuuid.uuid()}"