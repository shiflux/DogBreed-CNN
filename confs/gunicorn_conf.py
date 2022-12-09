from multiprocessing import cpu_count



# Socket Path

bind = '127.0.0.1:8000'



# Worker Options

workers = cpu_count() + 1

worker_class = 'uvicorn.workers.UvicornWorker'



# Logging Options

loglevel = 'info'

accesslog = '/var/log/access_log'

errorlog =  '/var/log/error_log'